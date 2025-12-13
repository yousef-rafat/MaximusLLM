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

INDEX = 0
TORCH_COMPILE = False
LONG_CONTEXT_TRAINING = False
MAX_LENGTH = 4096 if not LONG_CONTEXT_TRAINING else 32768
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 1_500_000
SAVE_EVERY_STEP = 10_000
QAT_TRAINING = False
TOTAL_NUMBER_OF_STEPS = 100_000
class Settings:
    lr_rate = 5e-5
    weight_decay = 0.1
    batch_size = 2

class CUDAPreFetch:
    def __init__(self, iter: IterableDataset, device: torch.device):
        self.iter = iter
        self.device = device
        self.current_stream = torch.cuda.Stream()
        self.next_batch = None

    def async_load(self):

        try:
            if not hasattr(self, 'iterator'):
                self.iterator = iter(self.iter)

            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.current_stream):
            self.next_batch = self.move(batch)
        
    def move(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking = True)
        elif isinstance(x, (list, tuple)):
            return type(x)(self.move(t) for t in x)
        elif isinstance(x, dict):
            return {k: self.move(v) for k, v in x.items()}
        else:
            return x
    
    def next(self):
        # wait for the last batch to finish loading
        torch.cuda.current_stream().wait_stream(self.current_stream)
        batch = self.next_batch

        self.async_load()
        return batch

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
                tokens = tokens[:MAX_LENGTH  - 1]
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

def compute_clm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index = -100
    )

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

    dataset = HFStreamDataset(world_size=world_size, rank = local_rank)
    config = Config.from_pretrained("google/gemma-3-270m", token = "")

    config.rope_theta = ROPE_THETA
    config.context_length = MAX_LENGTH
    
    model = Model(config, device).to(torch.bfloat16)
    prefetch = CUDAPreFetch(dataset, device)
    prefetch.async_load()

    model = torch.nn.parallel.DistributedDataParallel(model, output_device = local_rank, device_ids = [local_rank])
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


    model.train()
    if TORCH_COMPILE:
        model = model.to(device)
        model = torch.compile(model, fullgraph = True, mode = "default")
    else:
        model = model.to(device)

    step = 0
    # TODO: check if it skips the first batch
    batch = prefetch.next()#.next()


    while batch is not None:
        inputs = batch
        next_batch = prefetch.next()

        with torch.amp.autocast(enabled = True, device_type = f"cuda:{local_rank}", dtype=torch.bfloat16):
            input_ids, attention_mask = inputs
            logits = model(input_ids, attention_mask = attention_mask)
            loss = compute_clm_loss(logits, input_ids)
        
        main_optimizer.zero_grad(set_to_none = True)
        second_optimizer.zero_grad(set_to_none = True)

        scaler.scale(loss).backward()

        scaler.step(main_optimizer)
        muon_scheduler.step()
        scaler.step(second_optimizer)
        adam_scheduler.step()
        scaler.update()

        if local_rank == 0 and step % 50 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f}")

        del input_ids, attention_mask, logits, loss
        torch.cuda.synchronize(device)

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
