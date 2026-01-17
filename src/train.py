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
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from utils import update_model_hf, get_raw_model

losses = []
INDEX = 3490 * torch.cuda.device_count()
TORCH_COMPILE = True
LONG_CONTEXT_TRAINING = False
MAX_LENGTH = 2048 if not LONG_CONTEXT_TRAINING else 32768
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 1_500_000
SAVE_EVERY_STEP = 10_000
QAT_TRAINING = False
TOTAL_NUMBER_OF_STEPS = 1_984
ACCUM_STEPS = 32
PACKING = True
WARMUP = TOTAL_NUMBER_OF_STEPS * 0.05
DECAY = TOTAL_NUMBER_OF_STEPS * 0.1
STABLE = TOTAL_NUMBER_OF_STEPS - (WARMUP + DECAY)


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
    weight_decay = 0.1
    batch_size = 8
    muon_lr = 0.01
    adamw_rate = 5e-4


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
        dataset: str = "HuggingFaceFW/fineweb",
        take=None,
        rank=0,
        world_size=1,
        max_retries=5,
    ):
        self.rank = rank
        self.world_size = world_size

        for _ in range(0, max_retries):
            try:
                dataset = load_dataset(
                    dataset, name="CC-MAIN-2024-10", split="train", streaming=True
                )
                break
            except:  # noqa: E722
                continue
        # TODO: better solution than islice (that doesn't send many HTTP requests)
        self.dataset = islice(dataset, INDEX * Settings.batch_size, None)
        self.tokenizer = AutoTokenizer.from_pretrained("yousefg/MaximusLLM")
        self.eos_token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        if take is not None:
            self.dataset = self.dataset.take(take)

    def __iter__(self, batch_size=Settings.batch_size):
        buffer = []
        batch = []
        attention_masks = []

        def return_batch(batch, attention_masks=None):
            if LONG_CONTEXT_TRAINING or PACKING:
                if isinstance(batch, list):
                    batch = torch.stack(batch)
                else:
                    batch = torch.tensor(batch, dtype=torch.long)
                yield batch, torch.ones_like(batch)
            else:
                yield (
                    pad_sequence(batch, batch_first=True),
                    pad_sequence(attention_masks, batch_first=True, padding_value=0),
                )

        for i, item in enumerate(self.dataset):
            if i % self.world_size != self.rank:
                continue

            tokens = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"]

            if PACKING:
                tokens.append(self.eos_token)
                buffer.extend(tokens)

                while len(buffer) >= MAX_LENGTH:
                    seq = buffer[:MAX_LENGTH]
                    buffer = buffer[MAX_LENGTH:]
                    batch.append(torch.tensor(seq, dtype=torch.long))

                    if len(batch) == batch_size:
                        yield from return_batch(batch)
                        batch = []
            # TODO: should mix long, short, and medium sequence lengths
            elif LONG_CONTEXT_TRAINING:
                # random start if longer than MAX_LENGTH
                if len(tokens) > MAX_LENGTH:
                    start = random.randint(0, len(tokens) - (MAX_LENGTH - 1))
                    tokens = tokens[start : start + (MAX_LENGTH - 1)]
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
                tokens = tokens[: MAX_LENGTH - 2]
                tokens += [self.eos_token]
                batch.append(torch.tensor(tokens))
                attention_masks.append(torch.ones(len(tokens)))

            # yield full batches
            if len(batch) == batch_size:
                yield from return_batch(batch, attention_masks)
                batch = []
                attention_masks = []

        # remaining tokens
        if buffer and not PACKING:
            batch.append(buffer[:MAX_LENGTH])
        elif (
            PACKING and buffer
        ):  # having consistent shapes -> better memory usuage with torch.compile
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

        # last partial batch
        if batch:
            if PACKING and attention_masks:
                while len(attention_masks) < len(batch):
                    attention_masks.insert(0, torch.ones(MAX_LENGTH))
            yield from return_batch(batch, attention_masks)


def perplexity(loss):
    nll_loss = loss.mean()
    return torch.exp(nll_loss)


def filter_ckpt_for_muon(ckpt, weight_decay=Settings.weight_decay):
    muon_param, adamw_param = [], []
    for name, param in ckpt.named_parameters():
        if param.dim() < 2:
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
    prefetch = CUDAPreFetch(dataset, device)
    prefetch.async_load()
    model.gradient_checkpointing = True
    criterion = LigerFusedLinearCrossEntropyLoss()

    param_groups = filter_ckpt_for_muon(model, Settings.weight_decay)

    # Split the groups for each optimizer
    adamw_groups = [group for group in param_groups if group["params"][0].dim() < 2]
    muon_groups = [group for group in param_groups if group["params"][0].dim() >= 2]

    main_optimizer = torch.optim.Muon(muon_groups, lr=Settings.muon_lr)
    second_optimizer = torch.optim.AdamW(adamw_groups, lr=Settings.adamw_rate)
    scaler = torch.amp.GradScaler(enabled=False)

    muon_scheduler = lr_scheduler_fn(main_optimizer)
    adam_scheduler = lr_scheduler_fn(second_optimizer)

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
        new_state_dict["lm_head.weight"] = new_state_dict["embed_tokens.weight"]
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
                loss = criterion(
                    model.module.lm_head.weight,
                    logits.view(-1, model.module.config.hidden_size),
                    input_ids.long().reshape(-1),
                )

                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(main_optimizer)
            scaler.unscale_(second_optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(main_optimizer)
            scaler.step(second_optimizer)
            scaler.update()

            main_optimizer.zero_grad(set_to_none=True)
            second_optimizer.zero_grad(set_to_none=True)

            muon_scheduler.step()
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

        if step == TOTAL_NUMBER_OF_STEPS:
            break

    dist.barrier()

    if local_rank == 0:
        save_model(get_raw_model(model), "model.safetensors")
        update_model_hf(os.path.abspath("model.safetensors"), token="")
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
