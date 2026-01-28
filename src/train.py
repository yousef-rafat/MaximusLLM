import os
import torch
import random
from model import Model, Config
from itertools import islice
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
import torch.ao.quantization as quant
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_model
from contextlib import nullcontext
from utils import update_model_hf, get_raw_model
from torch.utils.data import get_worker_info
import torch.nn.functional as F
from huggingface_hub import list_repo_files

from transformers.utils import logging
logging.set_verbosity_error()

try: 
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:
    pass

losses = []
INDEX = 3490 * torch.cuda.device_count()
TORCH_COMPILE = True
USE_FAST_SOFTMAX = False
LONG_CONTEXT_TRAINING = False
MAX_LENGTH = 2048 if not LONG_CONTEXT_TRAINING else 32768
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 1_500_000
SAVE_EVERY_STEP = 10_000
QAT_TRAINING = False
ACCUM_STEPS = 32
TOTAL_NUMBER_OF_STEPS = 1_984 // ACCUM_STEPS
PACKING = True
WARMUP = max(30, TOTAL_NUMBER_OF_STEPS * 0.05) # stability
DECAY = TOTAL_NUMBER_OF_STEPS * 0.1
STABLE = TOTAL_NUMBER_OF_STEPS - (WARMUP + DECAY)

# hook: frequency_in_steps
TRAINING_HOOKS = {"balance_svd_layers": 1}


def lr_scheduler_fn(optimizer, min_lr=0.1):
    def scheduler(current_step):
        if current_step < WARMUP:
            return float(current_step) / float(WARMUP)
        if current_step < (WARMUP + STABLE):
            return 1.0
        decay_step = current_step - (STABLE + WARMUP)
        decay_progress = float(decay_step) / float(DECAY)
        return max(min_lr, 1 - decay_progress)  # avoids 0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)


class Settings:
    weight_decay = 0.05
    batch_size = 8
    muon_lr = 0.002
    adamw_rate = 4e-4
    use_adamw_only = False


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


def diagonal_attn_mask(seq, eos_token):
    T = len(seq)
    is_eos = seq == eos_token
    doc_ids = is_eos.cumsum(0).roll(1)
    doc_ids[0] = 0

    row = doc_ids.unsqueeze(0).expand(T, T)
    col = doc_ids.unsqueeze(1).expand(T, T)
    diff_doc_mask = row != col

    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    combined_mask = diff_doc_mask | causal_mask
    ids = torch.arange(len(seq), device=seq.device)
    combined_mask[ids, ids] = False
    return combined_mask


class HFStreamDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb",
        subset = "CC-MAIN-2024-10",
        take=None,
        rank=0,
        world_size=1,
        files_to_skip=0
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.take = take
        self.rank = rank
        self.world_size = world_size
        self.files_to_skip = files_to_skip

        self.tokenizer = AutoTokenizer.from_pretrained("yousefg/MaximusLLM")
        self.eos_token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        try:
            self.all_files = sorted(list_repo_files(
                dataset_name, 
                repo_type="dataset",
                allow_patterns=f"data/{subset}/*.parquet"
            ))
            self.all_file_urls = [
                f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{f}" 
                for f in self.all_files
            ]
        except Exception as e:
            print(f"Error fetching files: {e}")
            self.all_file_urls = []

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        global_num_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        # we shard the file urls between multiple workers
        my_urls = self.all_file_urls[global_worker_id::global_num_workers]

        if self.files_to_skip > 0:
            print(f"[Rank {self.rank}] Skipping first {self.files_to_skip} files.")
            my_urls = my_urls[self.files_to_skip:]

        if not my_urls:
            print(f"[Rank {self.rank} Worker {worker_id}] No files left to process.")
            return

        print(f"[Rank {self.rank} Worker {worker_id}] Streaming {len(my_urls)} files...")

        ds = load_dataset(
            "parquet", 
            data_files=my_urls, 
            split="train", 
            streaming=True
        )

        if self.take:
            ds = ds.take(self.take)

        buffer = []
        batch = []
        attention_masks = []

        def return_batch(batch, attention_masks=None):
            if LONG_CONTEXT_TRAINING or PACKING:
                if isinstance(batch, list):
                    b = torch.stack(batch)
                else:
                    b = torch.tensor(batch, dtype=torch.long)
                yield b, torch.ones_like(b)
            else:
                yield (
                    pad_sequence(batch, batch_first=True),
                    pad_sequence(attention_masks, batch_first=True, padding_value=0),
                )

        for item in ds:
            tokens = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"]

            if PACKING:
                tokens.append(self.eos_token)
                buffer.extend(tokens)

                while len(buffer) >= MAX_LENGTH:
                    seq = buffer[:MAX_LENGTH]
                    buffer = buffer[MAX_LENGTH:]
                    batch.append(torch.tensor(seq, dtype=torch.long))

                    if len(batch) == Settings.batch_size:
                        yield from return_batch(batch)
                        batch = []
            
            elif LONG_CONTEXT_TRAINING:
                if len(tokens) > MAX_LENGTH:
                    start = random.randint(0, len(tokens) - (MAX_LENGTH - 1))
                    tokens = tokens[start : start + (MAX_LENGTH - 1)]
                    seq = tokens + [self.eos_token]
                    batch.append(seq)
                else:
                    buffer.extend(tokens)
                    if buffer:
                        buffer.append(self.eos_token)
                    while len(buffer) >= MAX_LENGTH:
                        seq = buffer[:MAX_LENGTH]
                        buffer = buffer[MAX_LENGTH:]
                        batch.append(seq)
            else:
                tokens = tokens[: MAX_LENGTH - 2]
                tokens += [self.eos_token]
                batch.append(torch.tensor(tokens))
                attention_masks.append(torch.ones(len(tokens)))

            if len(batch) == Settings.batch_size:
                yield from return_batch(batch, attention_masks)
                batch = []
                attention_masks = []

        if buffer and not PACKING:
            batch.append(buffer[:MAX_LENGTH])
        elif PACKING and buffer:
            seq = torch.tensor(buffer, dtype=torch.long)
            pad_len = MAX_LENGTH - len(seq)
            if pad_len > 0:
                mask = torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
                seq = torch.nn.functional.pad(
                    seq, (0, pad_len), value=self.tokenizer.pad_token_id or 0
                )
                batch.append(seq)
                attention_masks = [mask]
            else:
                batch.append(seq)
                attention_masks = []

        if batch:
            if PACKING and attention_masks:
                while len(attention_masks) < len(batch):
                    attention_masks.insert(0, torch.ones(MAX_LENGTH))
            yield from return_batch(batch, attention_masks)

# take target embeddings + random embeddings to compute the softmax
# instead of expensive softmax over 260k vocab size
def fast_sampled_softmax_loss(hidden_states, target_ids, embedding_weight, n_negatives=4096):
    batch_size, hidden_dim = hidden_states.shape
    vocab_size = embedding_weight.shape[0]

    scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32, device=hidden_states.device))
    
    target_embeddings = F.embedding(target_ids, embedding_weight)
    
    true_logits = (hidden_states * target_embeddings).sum(dim=1, keepdim=True) / scale

    random_neg_ids = torch.randint(0, vocab_size, (n_negatives,), device=hidden_states.device)
    random_neg_vectors = F.embedding(random_neg_ids, embedding_weight)
    
    random_neg_logits = torch.matmul(hidden_states, random_neg_vectors.t()) / scale
    collision_mask = target_ids.unsqueeze(1) == random_neg_ids.unsqueeze(0)
    random_neg_logits = random_neg_logits.masked_fill(collision_mask, float('-inf'))

    num_batch_neg = min(2048, batch_size)
    batch_neg_indices = torch.randperm(batch_size, device=hidden_states.device)[:num_batch_neg]
    batch_neg_vectors = target_embeddings[batch_neg_indices]
    
    batch_neg_indices = target_ids[batch_neg_indices]
    batch_neg_logits = torch.matmul(hidden_states, batch_neg_vectors.t()) / scale

    collision_mask = target_ids.unsqueeze(1) == batch_neg_indices.unsqueeze(0)
    batch_neg_logits = batch_neg_logits.masked_fill(collision_mask, float("-inf"))

    all_logits = torch.cat([true_logits, random_neg_logits, batch_neg_logits], dim=1)

    # index 0 == target
    pseudo_labels = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
    
    return F.cross_entropy(all_logits, pseudo_labels)

def perplexity(loss):
    nll_loss = loss.mean()
    return torch.exp(nll_loss)


def filter_ckpt_for_muon(ckpt, weight_decay=Settings.weight_decay):
    muon_param, adamw_param = [], []
    for name, param in ckpt.named_parameters():
        if param.dim() < 2 or "embed" in name or "lm_head" in name:
            adamw_param.append(param)
        else:
            muon_param.append(param)

    return [
        {"params": adamw_param, "weight_decay": 0.0},
        {"params": muon_param, "weight_decay": weight_decay},
    ]


def main(local_rank, world_size):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float16

    dataset = HFStreamDataset(world_size=world_size, rank=local_rank)
    config = Config.from_pretrained("yousefg/MaximusLLM")

    config.rope_theta = ROPE_THETA
    config.context_length = MAX_LENGTH
    config.use_lora = LONG_CONTEXT_TRAINING

    # for muons stability, we init the model to fp32
    model = Model(config, device).float()
    model.to(device)
    dataset = torch.utils.data.DataLoader(
        dataset, num_workers = 4, prefetch_factor = 2, batch_size=None, pin_memory = True
    )

    prefetch = CUDAPreFetch(dataset, device)
    prefetch.async_load()
    model.gradient_checkpointing = True
    if not USE_FAST_SOFTMAX:
        criterion = LigerFusedLinearCrossEntropyLoss()

    param_groups = filter_ckpt_for_muon(model, Settings.weight_decay)

    # Split the groups for each optimizer
    adamw_groups = [group for group in param_groups if group["params"][0].dim() < 2]
    muon_groups = [group for group in param_groups if group["params"][0].dim() >= 2]

    if not Settings.use_adamw_only:
        main_optimizer = torch.optim.Muon(muon_groups, lr=Settings.muon_lr)
        muon_scheduler = lr_scheduler_fn(main_optimizer)
    else:
        adamw_groups += muon_groups

    second_optimizer = torch.optim.AdamW(adamw_groups, lr=Settings.adamw_rate)
    adam_scheduler = lr_scheduler_fn(second_optimizer)

    scaler = torch.amp.GradScaler(enabled=True)


    if os.path.exists("model.safetensors"):
        try:
            checkpoint = torch.load("model.safetensors")
            model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(e)
    else:
        checkpoint = load_file(
            hf_hub_download(
                repo_id="yousefg/MaximusLLM",
                filename="model.safetensors",
                local_dir=".",
            )
        )
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("model."):
                new_key = key[6:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
        del new_state_dict, checkpoint

    model.train()
    if TORCH_COMPILE:
        model = model.to(device)
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    else:
        model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        output_device=local_rank,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )

    # quantization
    if QAT_TRAINING:
        # fbgemm for x86 | qnnpack = ARM/mobile
        q_config = quant.get_default_qat_qconfig("fbgemm")
        model.qconfig = q_config
        quant.prepare_qat(model, inplace=True)

    step = 0
    # TODO: check if it skips the first batch
    batch = prefetch.next()  # .next()

    while batch is not None:
        inputs = batch
        next_batch = prefetch.next()

        is_update_step = ((step + 1) % ACCUM_STEPS == 0) or (next_batch is None)
        sync_context = model.no_sync() if not is_update_step else nullcontext()

        with sync_context:
            with torch.amp.autocast(
                enabled=True, device_type=f"cuda:{local_rank}", dtype=dtype
            ):
                input_ids, attention_mask = inputs
                # safety
                if input_ids.max().item() > model.module.embed_tokens.weight.shape[0]:
                    print("skipping batch")
                    step += 1
                    if step == TOTAL_NUMBER_OF_STEPS:
                        break
                    continue
                logits = model(
                    input_ids, attention_mask=attention_mask, return_hidden=True
                )

                logits = logits[:, :-1, :].reshape(-1, model.module.config.hidden_size)
                input_ids = input_ids[:, 1:].long().reshape(-1)

                if USE_FAST_SOFTMAX:
                    loss = fast_sampled_softmax_loss(logits, input_ids, model.module.lm_head.weight)
                else:
                    loss = criterion(
                        model.module.lm_head.weight,
                        logits,
                        input_ids,
                    )

                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            if not Settings.use_adamw_only:
                scaler.unscale_(main_optimizer)
            scaler.unscale_(second_optimizer)

            for k, v in TRAINING_HOOKS.items():
                if hasattr(model.module, k):
                    div_steps = v
                    if div_steps == 0:
                        continue
                    attr = getattr(model.module, k)
                    if (((step + 1) // ACCUM_STEPS) % div_steps) == 0:
                        attr()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            if not Settings.use_adamw_only:
                scaler.step(main_optimizer)
            scaler.step(second_optimizer)
            scaler.update()

            if not Settings.use_adamw_only:
                main_optimizer.zero_grad(set_to_none=True)
                muon_scheduler.step()

            second_optimizer.zero_grad(set_to_none=True)
            adam_scheduler.step()

            if local_rank == 0 and ((step + 1) // ACCUM_STEPS) % 10 == 0:
                print(
                    f"step {(step // ACCUM_STEPS):05d} | loss {(loss.item() * ACCUM_STEPS):.4f}"
                )
                losses.append(loss.detach().cpu())
            

        del input_ids, attention_mask, logits, loss

        batch = next_batch
        step += 1

        if step % SAVE_EVERY_STEP == 0 and local_rank == 0:
            save_model(get_raw_model(model), f"model_{step}.safetensors")

        if step == (TOTAL_NUMBER_OF_STEPS * ACCUM_STEPS):
            break

    dist.barrier()

    if local_rank == 0:
        save_model(get_raw_model(model), "model.safetensors")
        update_model_hf(os.path.abspath("model.safetensors"))
        print("model saved")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    plt.plot(losses)
    plt.savefig("loss_fig.png")
