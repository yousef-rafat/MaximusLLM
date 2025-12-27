import os
import torch
import random
from model import Model, Config
from itertools import islice
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
import torch.ao.quantization as quant
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

losses = []
INDEX = 0
TORCH_COMPILE = False
LONG_CONTEXT_TRAINING = False
MAX_LENGTH = 4096 if not LONG_CONTEXT_TRAINING else 32768
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 1_500_000
SAVE_EVERY_STEP = 10_000
QAT_TRAINING = False
TOTAL_NUMBER_OF_STEPS = 100_000
ACCUM_STEPS = 128
class Settings:
    lr_rate = 5e-5
    weight_decay = 0.1
    batch_size = 8

class CUDAPreFetch:
    def __init__(self, iter_, device):
        self.iter = iter(iter_)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.async_load()

    def move(self, x):
        if torch.is_tensor(x):
            return x.pin_memory()
        elif isinstance(x, (list, tuple)):
            return type(x)(self.move(t) for t in x)
        elif isinstance(x, dict):
            return {k: self.move(v) for k, v in x.items()}
        else:
            return x

    def async_load(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        self.next_batch = self.move(batch)

    def _to_device_async(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=True)
        elif isinstance(x, (list, tuple)):
            return type(x)(self._to_device_async(t) for t in x)
        elif isinstance(x, dict):
            return {k: self._to_device_async(v) for k, v in x.items()}
        else:
            return x

    def next(self):
        batch = self.next_batch
        if batch is None:
            return None
        with torch.cuda.stream(self.stream):
            batch_gpu = self._to_device_async(batch)
        torch.cuda.current_stream().wait_stream(self.stream)
        self.async_load()
        return batch_gpu

class HFStreamDataset(IterableDataset):
    def __init__(self, dataset: str = "HuggingFaceFW/fineweb", take = None, rank = 0, world_size = 1):

        self.rank = rank
        self.world_size = world_size

        dataset = load_dataset(dataset, name = "CC-MAIN-2024-10", split = "train", streaming = True)
        self.dataset = islice(dataset, INDEX, None)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", token = "") # requires a token
        self.eos_token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        if take is not None:
            self.dataset = self.dataset.take(take)

    def __iter__(self, batch_size = Settings.batch_size):
        buffer = []
        batch = []
        attention_masks = []

        def return_batch(batch, attention_masks = None):
            if LONG_CONTEXT_TRAINING:
                yield torch.tensor(batch, dtype=torch.long), torch.ones_like(batch)
            else:
                yield pad_sequence(batch, batch_first = True), pad_sequence(attention_masks, batch_first = True, padding_value = 0)

        for i, item in enumerate(self.dataset):

            if i % self.world_size != self.rank:
                continue

            tokens = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"]

            # TODO: should mix long, short, and medium sequence lengths
            if LONG_CONTEXT_TRAINING:
                # random start if longer than MAX_LENGTH
                if len(tokens) > MAX_LENGTH:
                    start = random.randint(0, len(tokens) - (MAX_LENGTH - 1))
                    tokens = tokens[start:start + (MAX_LENGTH - 1)]
                    seq = tokens + [self.eos_token]
                    batch.append(seq)
                else:
                    # accumulate tokens in a buffer
                    buffer.extend(tokens)

                    if buffer:
                        buffer.append(self.eos_token)

                    while len(buffer) >= MAX_LENGTH:
                        seq = buffer[:MAX_LENGTH]
                        buffer = buffer[MAX_LENGTH:]
                        batch.append(seq)
            else:
                tokens = tokens[:MAX_LENGTH  - 2]
                tokens += [self.eos_token]
                batch.append(torch.tensor(tokens))
                attention_masks.append(torch.ones(len(tokens)))

            # yield full batches
            if len(batch) == batch_size:
                yield from return_batch(batch, attention_masks)
                batch = []
                attention_masks = []

        # remaining tokens
        if buffer:
            batch.append(buffer[:MAX_LENGTH])

        # last partial batch
        if batch:
            yield from return_batch(batch, attention_masks)

# cce + gradient filtering
class SparseCCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, W, target, chunk_size=4096):
        ctx.save_for_backward(hidden_states, W, target)
        ctx.chunk_size = chunk_size
        
        relevant_weights = W[target]
        target_logits = torch.sum(hidden_states * relevant_weights, dim=1)
        
        N = hidden_states.size(0)
        max_scores = torch.full((N,), float('-inf'), device=hidden_states.device, dtype=torch.float32)
        sum_exp = torch.zeros(N, device=hidden_states.device, dtype=torch.float32)
        
        vocab_size = W.size(0)
        
        for i in range(0, vocab_size, chunk_size):
            end = min(i + chunk_size, vocab_size)
            W_chunk = W[i:end].T
            
            logits_chunk = torch.matmul(hidden_states, W_chunk)
            logits_chunk = logits_chunk.float() # ensure higher precision for exp
            
            chunk_max, _ = torch.max(logits_chunk, dim=1)
            new_max = torch.max(max_scores, chunk_max)
            
            exp_diff = torch.exp(max_scores - new_max)
            chunk_sum = torch.sum(torch.exp(logits_chunk - new_max.unsqueeze(1)), dim=1)
            sum_exp = sum_exp * exp_diff + chunk_sum
            max_scores = new_max
            
        lse = max_scores + torch.log(sum_exp)
        ctx.lse = lse
        ctx.valid_tokens = (target != -100).sum()
        
        return (lse - target_logits.float()).sum() / ctx.valid_tokens

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, W, target = ctx.saved_tensors
        lse = ctx.lse
        chunk_size = ctx.chunk_size
        valid_tokens = ctx.valid_tokens
        
        grad_hidden = torch.zeros_like(hidden_states)
        grad_W = torch.zeros_like(W)
        grad_scale = (grad_output / valid_tokens).float()
        
        relevant_weights = W[target]
        grad_hidden.add_(relevant_weights, alpha=-grad_scale)
        grad_W.index_add_(0, target, hidden_states, alpha=-grad_scale.to(hidden_states.dtype))

        vocab_size = W.size(0)

        sparsity_threshold = 10.0 
        
        min_lse = lse.min()
        
        for i in range(0, vocab_size, chunk_size):
            end = min(i + chunk_size, vocab_size)
            W_chunk = W[i:end].T
            
            logits_chunk = torch.matmul(hidden_states, W_chunk)
            
            chunk_max_val = logits_chunk.max()
            
            if chunk_max_val < (min_lse - sparsity_threshold):
                continue

            softmax_chunk = torch.exp(logits_chunk.float() - lse.unsqueeze(1)) * grad_scale
            softmax_chunk = softmax_chunk.to(hidden_states.dtype)

            grad_hidden.addmm_(softmax_chunk, W_chunk.T)
            grad_W[i:end].addmm_(softmax_chunk.T, hidden_states)
            
        return grad_hidden, grad_W, None, None

def cce_loss(hidden, W, targets, chunk_size=4096):
    h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    t = targets[:, 1:].reshape(-1)
    mask = t != -100
    h = h[mask]
    t = t[mask]
    return SparseCCE.apply(h, W, t, chunk_size)

def perplexity(loss):
    nll_loss = loss.mean()
    return torch.exp(nll_loss)

def filter_ckpt_for_muon(ckpt, weight_decay = Settings.weight_decay):
    muon_param, adamw_param = [], []
    for name, param in ckpt.named_parameters():
        if param.dim() < 2:
            adamw_param.append(param)
        else:
            muon_param.append(param)

    return [
        {"params": adamw_param, "weight_decay": 0.0},
        {"params": muon_param, "weight_decay": weight_decay}
    ]
    
            
def main(local_rank, world_size):

    dist.init_process_group(backend='nccl', world_size = world_size, rank = local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float16

    dataset = HFStreamDataset(world_size=world_size, rank = local_rank)
    config = Config.from_pretrained("google/gemma-3-270m", token = "")

    config.rope_theta = ROPE_THETA
    config.context_length = MAX_LENGTH
    
    # for muons stability, we init the model to fp32
    model = Model(config, device).float()
    prefetch = CUDAPreFetch(dataset, device)
    prefetch.async_load()

    model.train()
    if TORCH_COMPILE:
        model = model.to(device)
        model = torch.compile(model, fullgraph = True, mode = "default")
    else:
        model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, output_device = local_rank, device_ids = [local_rank], find_unused_parameters=True)
    param_groups = filter_ckpt_for_muon(model.module, Settings.weight_decay)
    
    # Split the groups for each optimizer
    adamw_groups = [group for group in param_groups if group['params'][0].dim() < 2]
    muon_groups = [group for group in param_groups if group['params'][0].dim() >= 2]

    main_optimizer = torch.optim.Muon(muon_groups, lr = Settings.lr_rate)
    second_optimizer = torch.optim.AdamW(adamw_groups, lr = Settings.lr_rate)
    scaler = torch.amp.GradScaler(enabled = False)

    muon_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_optimizer, TOTAL_NUMBER_OF_STEPS)
    adam_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(second_optimizer, TOTAL_NUMBER_OF_STEPS)

    if os.path.exists("model.pt"):
        try:
            checkpoint = torch.load("model.pt")
            model.load_state_dict(checkpoint, strict =  False)
        except Exception as e:
            print(e)

    # quantization
    if QAT_TRAINING:
        # fbgemm for x86 | qnnpack = ARM/mobile
        q_config = quant.get_default_qat_qconfig("fbgemm")
        model.qconfig = q_config
        quant.prepare_qat(model, inplace = True)


    step = 0
    # TODO: check if it skips the first batch
    batch = prefetch.next()#.next()


    while batch is not None:
        inputs = batch
        next_batch = prefetch.next()

        is_sync_step = ((step + 1) % ACCUM_STEPS == 0) or (next_batch is None)
        context = model.no_sync() if not is_sync_step else torch.enable_grad()
        
        with context:
            with torch.amp.autocast(enabled = True, device_type = f"cuda:{local_rank}", dtype=dtype):
                input_ids, attention_mask = inputs
                logits = model(input_ids, attention_mask = attention_mask, return_hidden=True)
                loss = cce_loss(logits, model.module.lm_head.weight, input_ids)

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            
            scaler.unscale_(main_optimizer)
            scaler.unscale_(second_optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(main_optimizer)
            scaler.step(second_optimizer)
            scaler.update()
            
            main_optimizer.zero_grad(set_to_none = True)
            second_optimizer.zero_grad(set_to_none = True)
            
            muon_scheduler.step()
            adam_scheduler.step()

            if local_rank == 0 and step % 1000 == 0:
                print(f"step {step:05d} | loss {loss.item():.4f}")
                losses.append(loss.detach().cpu())

        del input_ids, attention_mask, logits, loss

        batch = next_batch
        step += 1

        if step % SAVE_EVERY_STEP == 0 and local_rank == 0:
            torch.save(model.state_dict(), f"model_{step}.pt")

        if step == TOTAL_NUMBER_OF_STEPS:
            break

    dist.destroy_process_group()
    
    if local_rank == 0:
        torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    plt.plot(losses)
    plt.show()
