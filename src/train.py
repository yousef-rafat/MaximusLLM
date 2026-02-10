import os
import torch
import random
import fnmatch
from model import Model, Config
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
import torch.ao.quantization as quant
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_model
from contextlib import nullcontext
from torch.utils.data import get_worker_info
from itertools import islice
import torch.nn.functional as F
from huggingface_hub import list_repo_files
from utils import update_model_hf, get_global_loss, get_raw_model

from transformers.utils import logging
logging.set_verbosity_error()

try: 
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:
    pass

losses = []
INDEX = 1_984
TORCH_COMPILE = True
USE_FAST_SOFTMAX = True
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
    batch_size = 12
    muon_lr = 0.002
    adamw_rate = 4e-4
    use_adamw_only = False
    aux_loss_percent = 0.2
    loss_ema_beta = 0.95
    matryoshka_scale = 20.0


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


def get_packed_mask_and_pos_ids(input_ids, eos_token_id):
    B, T = input_ids.shape
    device = input_ids.device

    is_eos = (input_ids == eos_token_id)

    doc_ids = is_eos.cumsum(dim=1).roll(shifts=1, dims=1)
    doc_ids[:, 0] = 0
    
    seq_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    
    first_token_mask = torch.zeros_like(is_eos)
    first_token_mask[:, 1:] = is_eos[:, :-1]
    
    offsets = torch.where(first_token_mask, seq_idx, torch.zeros_like(seq_idx))
    offsets = offsets.cummax(dim=1)[0]
    position_ids = seq_idx - offsets

    row_doc = doc_ids.unsqueeze(2)
    col_doc = doc_ids.unsqueeze(1)
    
    doc_mask = (row_doc == col_doc)
    
    causal_mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))

    mask = doc_mask & causal_mask.unsqueeze(0)
    
    return mask, position_ids


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

        self.TOKENS_PER_FILE_EST = 500_000_000 
        self.ROWS_PER_FILE_EST = 300_000 

        tokens_processed = INDEX * Settings.batch_size * world_size * MAX_LENGTH

        tokens_per_worker = tokens_processed / world_size
        
        self.files_per_worker_skipped = int(tokens_per_worker // self.TOKENS_PER_FILE_EST)
        
        # how far in the file
        tokens_remainder = tokens_per_worker % self.TOKENS_PER_FILE_EST
        percent_done = tokens_remainder / self.TOKENS_PER_FILE_EST
        
        self.rows_to_skip = int(percent_done * self.ROWS_PER_FILE_EST)

        if INDEX > 0 and self.rank == 0:
            print(f"Resuming Step {INDEX} | Total Processed: {tokens_processed/1e9:.2f}B tokens")
            print(f"Skipping {self.files_per_worker_skipped} full files per worker.")
            print(f"Skipping first {self.rows_to_skip} rows in the active file.")

        try:
            all_repo_files = sorted(list_repo_files(
                dataset_name, 
                repo_type="dataset",
                token=None
            ))

            filter_pattern = f"data/{subset}/*.parquet"
            self.all_files = sorted([
                f for f in all_repo_files if fnmatch.fnmatch(f, filter_pattern)
            ])
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

        if self.files_per_worker_skipped > 0:
            if self.files_per_worker_skipped < len(my_urls):
                my_urls = my_urls[self.files_per_worker_skipped:]
            else:
                print(f"[Rank {self.rank}] Worker {worker_id} has finished all assigned files.")
                return

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

        def return_batch(batch):
            if isinstance(batch, list):
                b = torch.stack(batch)
            else:
                b = torch.tensor(batch, dtype=torch.long)
            yield b 

        ds_iterator = iter(ds)
        if self.rows_to_skip > 0:
            ds_iterator = islice(ds_iterator, self.rows_to_skip, None)

        for item in ds_iterator:
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

            if len(batch) == Settings.batch_size:
                yield from return_batch(batch)
                batch = []

        if buffer and not PACKING:
            batch.append(buffer[:MAX_LENGTH])
        elif PACKING and buffer:
            seq = torch.tensor(buffer, dtype=torch.long)
            pad_len = MAX_LENGTH - len(seq)
            if pad_len > 0:
                seq = torch.nn.functional.pad(
                    seq, (0, pad_len), value=self.tokenizer.pad_token_id or 0
                )
                batch.append(seq)
            else:
                batch.append(seq)

        if batch:
            yield from return_batch(batch)

class MatryoshkaManualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, embedding_weight, target_ids, 
                low_rank_dim, n_candidates, chunk_size, with_batch_mean, aux_weight):
        
        N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype # Capture original dtype (e.g. float16)
        
        scale = Settings.matryoshka_scale

        h_norm_val = hidden_states.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        h_full = hidden_states / h_norm_val
        h_low = h_full[:, :low_rank_dim]

        with torch.no_grad():
            w_low_norm = F.normalize(embedding_weight[:, :low_rank_dim], p=2, dim=-1)

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        saved_chunks = []

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            curr_h_f = h_full[i:end]
            curr_h_l = h_low[i:end]
            curr_t_ids = target_ids[i:end]

            with torch.no_grad():
                if with_batch_mean:
                    h_mean = curr_h_l.mean(dim=0, keepdim=True)
                    scan_logits = torch.matmul(h_mean, w_low_norm.t())
                    _, top_indices = torch.topk(scan_logits, n_candidates, dim=1)
                    top_indices = top_indices.squeeze(0)
                else:
                    scan_logits = torch.matmul(curr_h_l, w_low_norm.t())
                    _, top_indices = torch.topk(scan_logits, n_candidates, dim=1)

            w_f_pos = F.normalize(embedding_weight[curr_t_ids], p=2, dim=-1)
            w_f_cand = F.normalize(embedding_weight[top_indices], p=2, dim=-1)
            
            pos_sims = (curr_h_f * w_f_pos).sum(dim=-1, keepdim=True)
            
            if with_batch_mean:
                neg_sims = torch.matmul(curr_h_f, w_f_cand.t())
                is_target = (top_indices.unsqueeze(0) == curr_t_ids.unsqueeze(1))
            else:
                neg_sims = torch.bmm(curr_h_f.unsqueeze(1), w_f_cand.transpose(1, 2)).squeeze(1)
                is_target = (top_indices == curr_t_ids.unsqueeze(1))
            
            neg_sims = neg_sims.masked_fill(is_target, float('-inf'))
            logits_m = torch.cat([pos_sims, neg_sims], dim=1).float() * scale
            
            # log softmax for stability
            log_probs_m = F.log_softmax(logits_m, dim=-1)
            loss_m = -log_probs_m[:, 0].sum()

            #   aux --------------------------------
            w_l_pos = F.normalize(w_f_pos[:, :low_rank_dim], p=2, dim=-1)
            w_l_cand = F.normalize(w_f_cand[:, :low_rank_dim], p=2, dim=-1)
            
            low_pos = (curr_h_l * w_l_pos).sum(dim=-1, keepdim=True)
            if with_batch_mean:
                low_neg = torch.matmul(curr_h_l, w_l_cand.t())
            else:
                low_neg = torch.bmm(curr_h_l.unsqueeze(1), w_l_cand.transpose(1, 2)).squeeze(1)
            
            low_neg = low_neg.masked_fill(is_target, float('-inf'))
            logits_a = torch.cat([low_pos, low_neg], dim=1).float() * scale
            
            log_probs_a = F.log_softmax(logits_a, dim=-1)
            loss_a = -log_probs_a[:, 0].sum()

            total_loss += (loss_m + aux_weight * loss_a)

            # save softmax probs
            saved_chunks.append((
                curr_h_f, curr_t_ids, top_indices, 
                log_probs_m.exp().to(dtype), log_probs_a.exp().to(dtype), 
                w_f_pos, w_f_cand, w_l_pos, w_l_cand,
                logits_m, logits_a
            ))

        ctx.save_for_backward(hidden_states, embedding_weight, h_norm_val)
        ctx.aux_weight = aux_weight
        ctx.with_batch_mean = with_batch_mean
        ctx.low_rank_dim = low_rank_dim
        ctx.saved_chunks = saved_chunks
        ctx.scale = scale
        
        return total_loss / N

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, embedding_weight, h_norm_val = ctx.saved_tensors
        aux_weight = ctx.aux_weight
        with_batch_mean = ctx.with_batch_mean
        low_rank_dim = ctx.low_rank_dim
        scale = ctx.scale
        
        N, _ = hidden_states.shape
        total_tokens = N
        dtype = hidden_states.dtype
        
        grad_h_normalized = torch.zeros_like(hidden_states, dtype=torch.float32)
        grad_embed = torch.zeros_like(embedding_weight, dtype=torch.float32)
        grad_scale_accum = torch.tensor(0.0, device=grad_output.device, dtype=torch.float32)

        chunk_size = ctx.saved_chunks[0][0].shape[0]
        grad_embed_low = grad_embed[:, :low_rank_dim]

        for i, chunk in enumerate(ctx.saved_chunks):
            (h_f, t_ids, top_idx, p_m, p_a, w_fp, w_fc, w_lp, w_lc, l_m, l_a) = chunk
            
            # index 0 = label
            # CE gradient = (softmax - label)
            dz_m = p_m.float().clone()
            dz_m[:, 0] -= 1.0

            dz_m = dz_m * (grad_output.float() / total_tokens) * scale
            
            dz_a = p_a.float().clone()
            dz_a[:, 0] -= 1.0
            dz_a = dz_a * (grad_output.float() / total_tokens) * scale * aux_weight

            dz_m_dt = dz_m.to(dtype)
            dz_a_dt = dz_a.to(dtype)

            # main loss
            gh_m = dz_m_dt[:, :1] * w_fp
            if with_batch_mean:
                gh_m += torch.matmul(dz_m_dt[:, 1:], w_fc)
            else:
                gh_m += torch.bmm(dz_m_dt[:, 1:].unsqueeze(1), w_fc).squeeze(1)
            
            # part of aux loss
            gh_a_low = dz_a_dt[:, :1] * w_lp
            if with_batch_mean:
                gh_a_low += torch.matmul(dz_a_dt[:, 1:], w_lc)
            else:
                gh_a_low += torch.bmm(dz_a_dt[:, 1:].unsqueeze(1), w_lc).squeeze(1)
            
            gh_m[:, :low_rank_dim] += gh_a_low
            
            grad_h_normalized[i*chunk_size : i*chunk_size + h_f.shape[0]] = gh_m.float()

            h_f_float = h_f.float()
            grad_embed.index_add_(0, t_ids, dz_m[:, :1] * h_f_float)
            
            if with_batch_mean:
                grad_w_neg = torch.matmul(dz_m_dt[:, 1:].t(), h_f).float()
                grad_embed.index_add_(0, top_idx, grad_w_neg)
            else:
                grad_w_neg = torch.bmm(dz_m_dt[:, 1:].unsqueeze(2), h_f.unsqueeze(1))
                for b in range(h_f.shape[0]):
                    grad_embed.index_add_(0, top_idx[b], grad_w_neg[b].float())

            h_l_float = h_f_float[:, :low_rank_dim] 
            grad_embed_low.index_add_(0, t_ids, dz_a[:, :1] * h_l_float)
            
            # AUX LOSS -----------------------------------------

            if with_batch_mean:
                grad_w_low_neg = torch.matmul(dz_a_dt[:, 1:].t(), h_f[:, :low_rank_dim]).float()
                grad_embed_low.index_add_(0, top_idx, grad_w_low_neg)
            else:
                # per token
                for b in range(h_f.shape[0]):
                    g_w_l_b = dz_a_dt[b, 1:].unsqueeze(1) * h_f[b, :low_rank_dim].unsqueeze(0)
                    grad_embed_low.index_add_(0, top_idx[b], g_w_l_b.float())

            grad_scale_accum += (dz_m * l_m / scale).sum() + (dz_a * l_a / scale).sum()

        # l2 norm formula -> grad_raw = (grad_norm - h_norm * (h_norm . grad_norm)) / norm_val        
        h_full_recalc_f = (hidden_states / h_norm_val).float()
        
        dot = (grad_h_normalized * h_full_recalc_f).sum(dim=-1, keepdim=True)
        grad_h = (grad_h_normalized - h_full_recalc_f * dot) / h_norm_val.float()
 
        return grad_h.to(dtype), grad_embed.to(dtype), None, None, None, None, None, None, None

class MatryoshkaSampledSoftmaxLoss(torch.nn.Module):
    def __init__(self, embedding_weight, low_rank_dim=64, n_candidates=2048, chunk_size=32):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.low_rank_dim = low_rank_dim
        self.n_candidates = n_candidates
        self.chunk_size = chunk_size

    def forward(self, hidden_states, target_ids, with_batch_mean=True):
        return MatryoshkaManualFunction.apply(
            hidden_states, 
            self.embedding_weight, 
            target_ids, 
            self.low_rank_dim,
            self.n_candidates,
            self.chunk_size,
            with_batch_mean,
            Settings.aux_loss_percent
        )

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
    eos_token = dataset.eos_token
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

    param_groups = filter_ckpt_for_muon(model, Settings.weight_decay)

    adamw_groups = [param_groups[0]] 
    muon_groups = [param_groups[1]]

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
    
    if not USE_FAST_SOFTMAX:
        loss_fn = LigerFusedLinearCrossEntropyLoss()
    else:
        loss_fn = MatryoshkaSampledSoftmaxLoss(model.embed_tokens.weight)

    model.train()
    if TORCH_COMPILE:
        model = model.to(device)
        #model = torch.compile(model, fullgraph=True, mode="default")
        # avoid problems with custom autograd fn and compiling
        for i, block in enumerate(model.layers):
            model.layers[i] = torch.compile(block, mode="default", fullgraph=True)
    else:
        model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        output_device=local_rank,
        device_ids=[local_rank],
        find_unused_parameters=False,
        bucket_cap_mb=100,
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
    running_loss = 0
    smoothed_loss = None

    while batch is not None:
        inputs = batch
        next_batch = prefetch.next()

        is_update_step = ((step + 1) % ACCUM_STEPS == 0) or (next_batch is None)
        sync_context = model.no_sync() if not is_update_step else nullcontext()

        with sync_context:
            with torch.amp.autocast(
                enabled=True, device_type=f"cuda:{local_rank}", dtype=dtype
            ):
                input_ids = inputs
                attention_mask, position_ids = get_packed_mask_and_pos_ids(input_ids, eos_token)
                # safety
                if input_ids.max().item() > model.module.embed_tokens.weight.shape[0]:
                    print("skipping batch")
                    step += 1
                    if step == TOTAL_NUMBER_OF_STEPS:
                        break
                    continue
                logits = model(
                    input_ids, attention_mask=attention_mask, cache_position=position_ids, return_hidden=True
                )

                eos_mask = (input_ids == eos_token)            
                loss_mask = ~eos_mask[:, :-1].reshape(-1)

                logits = logits[:, :-1, :].reshape(-1, model.module.config.hidden_size)[loss_mask]
                input_ids = input_ids[:, 1:].long().reshape(-1)[loss_mask]

                if USE_FAST_SOFTMAX:
                    loss = loss_fn(logits, input_ids)
                else:
                    loss = loss_fn(
                        model.module.lm_head.weight,
                        logits,
                        input_ids,
                    )

                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
        
        running_loss += loss.detach().item()
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

            old_scale = scaler.get_scale()

            if not Settings.use_adamw_only:
                scaler.step(main_optimizer)
            scaler.step(second_optimizer)
            scaler.update()

            new_scale = scaler.get_scale()

            if new_scale >= old_scale:
                if not Settings.use_adamw_only:
                    muon_scheduler.step()
                adam_scheduler.step()
            else:
                # may remove this if it got too annoying
                if local_rank == 0:
                    print(f"Step {step} was skipped")

            if not Settings.use_adamw_only:
                main_optimizer.zero_grad(set_to_none=True)
            second_optimizer.zero_grad(set_to_none=True)

            true_loss = get_global_loss(running_loss, world_size)
            if smoothed_loss is None:
                smoothed_loss = true_loss
            smoothed_loss = Settings.loss_ema_beta * smoothed_loss + (1 - Settings.loss_ema_beta) * true_loss

            if local_rank == 0 and ((step + 1) // ACCUM_STEPS) % 10 == 0:
                print(
                    f"step {(step // ACCUM_STEPS):05d} | loss {(smoothed_loss):.4f}"
                )
                #losses.append(loss.detach().cpu())
            
            running_loss = 0
            

        del input_ids, attention_mask, logits, loss

        batch = next_batch
        step += 1

        # TODO
        if step % SAVE_EVERY_STEP == 0 and local_rank == 0:
            save_model(get_raw_model(model), f"model_{step}.safetensors")

        if step == (TOTAL_NUMBER_OF_STEPS * ACCUM_STEPS):
            break

    dist.barrier()

    if local_rank == 0:
        save_model(get_raw_model(model), "model.safetensors")
        if hasattr(os, 'sync'):
            os.sync()
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
