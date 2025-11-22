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

INDEX = 0
LONG_CONTEXT_TRAINING = False
MAX_LENGTH = 4096 if not LONG_CONTEXT_TRAINING else 32768
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 1_500_000
SAVE_EVERY_STEP = 10_000
QAT_TRAINING = False

class Settings:
    lr_rate = 5e-5
    weight_decay = 0.1
    batch_size = 16

class CUDAPreFetch:
    def __init__(self, iter: IterableDataset, device: torch.device):
        self.iter = iter
        self.device = device
        self.current_stream = torch.cuda.Stream()
        self.next_batch = None

    def async_load(self):

        try:
            batch = next(self.iter)
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
        self.tokenizer = AutoTokenizer.from_pretrained("")
        self.eos_token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        if take is not None:
            self.dataset = self.dataset.take(take)

    def __iter__(self, batch_size = Settings.batch_size):
        buffer = []
        batch = []

        for i, item in enumerate(self.dataset):

            if i % self.world_size != self.rank:
                continue

            tokens = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"]

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

            # yield full batches
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []

        # remaining tokens
        if buffer:
            batch.append(buffer[:MAX_LENGTH])

        # last partial batch
        if batch:
            yield torch.tensor(batch, dtype=torch.long)

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

def apply_weight_decay(model, weight_decay = 0.1):
    no_decay_params, decay_params = [], []
    for name, param in model.parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith("bias")
            or "embedding" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": weight_decay}
    ]
            
def main(local_rank, world_size):

    dist.init_process_group(backend='nccl', world_size = world_size, rank = local_rank)

    device = torch.device(f"cuda:{local_rank}")

    dataset = HFStreamDataset(world_size=world_size, rank = local_rank)
    config = Config(rope_theta=ROPE_THETA, context_length=MAX_LENGTH)
    
    model = Model(config)
    prefetch = CUDAPreFetch(dataset, device)

    optimizer = torch.optim.AdamW(apply_weight_decay(model, Settings.weight_decay), lr = Settings.lr_rate)
    scaler = torch.cuda.amp.GradScaler(enabled = True)

    if os.path.exists("model.pt"):
        checkpoint = torch.load("model.pt")
        model.load_state_dict(checkpoint)

    # quantization
    if QAT_TRAINING:
        # fbgemm for x86 | qnnpack = ARM/mobile
        q_config = quant.get_default_qat_qconfig("fbgemm")
        model.qconfig = q_config
        quant.prepare_qat(model, inplace = True)


    model.train()
    model = torch.compile(model).to(device)

    step = 0
    batch = prefetch.next()

    model = torch.nn.parallel.DistributedDataParallel(model, output_device = local_rank, device_ids = [local_rank])

    while batch is not None:
        inputs = batch
        next_batch = prefetch.next()

        with torch.cuda.amp.autocast(enabled = True):
            logits = model(inputs)
            loss = compute_clm_loss(logits, inputs)
        
        optimizer.zero_grad(set_to_none = True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if local_rank == 0 and step % 50 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f}")

        batch = next_batch
        step += 1

        if step % SAVE_EVERY_STEP == 0 and local_rank == 0:
            torch.save(model.state_dict(), f"model_{step}.pt")

    dist.destroy_process_group()
    
    if local_rank == 0:
        torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)