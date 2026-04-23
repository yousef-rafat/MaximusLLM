import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import random
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
from lora import blockswap_attention_layers

from transformers.utils import logging
logging.set_verbosity_error()

try: 
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:
    pass

losses = []
INDEX = 0
TORCH_COMPILE = True
USE_FAST_SOFTMAX = True
LONG_CONTEXT_TRAINING = True
MAX_LENGTH = 2048 if not LONG_CONTEXT_TRAINING else 8192
ROPE_THETA = 10_000 if not LONG_CONTEXT_TRAINING else 100_000
SAVE_EVERY_STEP = 110
QAT_TRAINING = False
ACCUM_STEPS = 32
TOTAL_NUMBER_OF_STEPS = 1_984 // ACCUM_STEPS
PACKING = True if not LONG_CONTEXT_TRAINING else False
WARMUP = max(30, TOTAL_NUMBER_OF_STEPS * 0.05) # stability
DECAY = TOTAL_NUMBER_OF_STEPS * 0.1
STABLE = TOTAL_NUMBER_OF_STEPS - (WARMUP + DECAY)
SKIP_BLOCK_MASK = True

# skip block mask bascially concat documents together with eos 
# while allowing attention to look at previous documents
if SKIP_BLOCK_MASK:
    LONG_CONTEXT_TRAINING = False
    PACKING = True

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
    batch_size = 2
    muon_lr = 0.002
    adamw_rate = 4e-4
    use_adamw_only = False
    aux_loss_percent = 0.2
    loss_ema_beta = 0.95
    random_slice_prob = 0.3
    SFT_TRAINING=False
    INSTRUCTION_TRAINING=True
    if INSTRUCTION_TRAINING:
        SFT_TRAINING = True


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
        dataset_name: str = "HuggingFaceH4/ultrachat_200k",
        subset = "train_sft",
        rank=0,
        world_size=1,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

        self.tokenizer = AutoTokenizer.from_pretrained("yousefg/MaximusLLM")
        self.eos_token = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        self.pad_token = self.tokenizer.pad_token_id

        AVG_ROWS_PER_SEQ = 8196 // 2048 # conservative
        APPROX_ROWS_PER_FILE = 15000 
        
        total_rows_global = INDEX * ACCUM_STEPS * (Settings.batch_size * world_size) * AVG_ROWS_PER_SEQ
        self.global_files_to_skip = int(total_rows_global // APPROX_ROWS_PER_FILE)
        print("skipping: ", total_rows_global)

        self.global_rows_to_skip_in_shard = int(total_rows_global % APPROX_ROWS_PER_FILE)

        try:
            all_repo_files = list_repo_files(dataset_name, repo_type="dataset")
            self.all_file_urls = sorted([
                f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{f}" 
                for f in all_repo_files if f.endswith(".parquet") and "data/" in f
            ])
            
            if self.global_files_to_skip < len(self.all_file_urls):
                self.remaining_urls = self.all_file_urls[self.global_files_to_skip:]
            else:
                self.remaining_urls = []
                
        except Exception as e:
            print(f"Error fetching files: {e}")
            self.remaining_urls = []

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        global_num_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id
        my_urls = self.remaining_urls[global_worker_id::global_num_workers]

        if not my_urls:
            return

        ds = load_dataset("parquet", data_files=my_urls, split="train", streaming=True)
        ds_iterator = iter(ds)

        if self.global_rows_to_skip_in_shard > 0:
            local_skip = self.global_rows_to_skip_in_shard // global_num_workers
            ds_iterator = islice(ds_iterator, local_skip, None)

        buffer = []
        batch = []

        batch_labels = []
        buffer_labels = []

        def return_batch(batch, labels=None):
            if isinstance(batch, list):
                if labels is None:
                    b = torch.stack(batch)
                else:
                    b = (torch.stack(batch), torch.stack(labels))
            else:
                if labels is None:
                    b = torch.tensor(batch, dtype=torch.long)
                else:
                    b = (torch.tensor(batch, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
            yield b 

        def return_padded_batch(batch, pad_token_id=self.pad_token):
            max_l = max([t.size(0) for t in batch])
            padded_batch = torch.full((len(batch), max_l), pad_token_id, dtype=torch.long)
            for i, t in enumerate(batch):
                padded_batch[i, :t.size(0)] = t
            return padded_batch


        def do_pack():
            nonlocal batch, batch_labels
            while len(buffer) >= MAX_LENGTH:
                seq = buffer[:MAX_LENGTH]

                del buffer[:MAX_LENGTH]
                batch.append(torch.tensor(seq, dtype=torch.long))

                if Settings.SFT_TRAINING:
                    labels = buffer_labels[:MAX_LENGTH]
                    del buffer_labels[:MAX_LENGTH]
                    batch_labels.append(torch.tensor(labels, dtype=torch.long))

                if len(batch) == Settings.batch_size:
                    if Settings.SFT_TRAINING:
                        yield from return_batch(batch, batch_labels)
                        batch, batch_labels = [], []
                    else:
                        yield from return_batch(batch)
                        batch = []

        for item in ds_iterator:
            if Settings.SFT_TRAINING:
                all_msg_tokens = []
                all_msg_labels = []

                for i, msg in enumerate(item["messages"]):
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "user":
                        text = f"<start_of_turn>user\n{content}<end_of_turn>\n"
                        msg_toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                        
                        all_msg_tokens.extend(msg_toks)
                        
                        if Settings.INSTRUCTION_TRAINING:
                            all_msg_labels.extend(msg_toks)
                        else:
                            all_msg_labels.extend([-100] * len(msg_toks))
                            
                    elif role == "model" or role == "assistant":
                        text = f"<start_of_turn>model\n{content}<end_of_turn>\n"
                        msg_toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                        
                        all_msg_tokens.extend(msg_toks)
                        all_msg_labels.extend(msg_toks)

                tokens = all_msg_tokens
                labels = all_msg_labels
            else:
                tokens = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"]
                tokens.append(self.eos_token)

            if PACKING and not LONG_CONTEXT_TRAINING:
                if Settings.SFT_TRAINING:
                    buffer_labels.extend(labels)

                buffer.extend(tokens)
                yield from do_pack()
            
            elif LONG_CONTEXT_TRAINING:
                # we mix between random slicing of bigger than 32k-length data and pushing full 32k samples
                # the rest variable length samples gets to buffer with packing logic
                full_tokens = tokens + [self.eos_token]
                doc_len = len(full_tokens)
                if len(batch) == 0:
                    current_target_len = random.choices([2048, MAX_LENGTH // 2, MAX_LENGTH], weights=[0.3, 0.3, 0.4], k=1)[0]

                if doc_len > current_target_len:
                    roll = random.random()
                    
                    # random 32k window from a large document
                    if roll < Settings.random_slice_prob:
                        start = random.randint(0, doc_len - current_target_len)
                        segment = full_tokens[start : start + current_target_len]
                        batch.append(torch.tensor(segment, dtype=torch.long))

                    else:
                        for j in range(0, doc_len, current_target_len):
                            chunk = full_tokens[j : j + current_target_len]
                            
                            if len(chunk) > 50:
                                batch.append(torch.tensor(chunk, dtype=torch.long))
                                if len(batch) == Settings.batch_size:
                                    yield return_padded_batch(batch)
                                    batch = []

                elif doc_len > (current_target_len * 0.75): # maximum of 25% padding
                    batch.append(torch.tensor(full_tokens, dtype=torch.long))

                if len(batch) == Settings.batch_size:
                    yield return_padded_batch(batch)
                    batch = []
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
                low_rank_dim, n_candidates, chunk_size, aux_weight):
        
        N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        h_f = hidden_states
        h_l = h_f[:, :low_rank_dim]
        w_low = embedding_weight[:, :low_rank_dim]

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        saved_chunks = []

        with torch.no_grad():
            w_norm_sq = (embedding_weight ** 2).sum(dim=-1).mean().item()
            w_low_norm_sq = (w_low ** 2).sum(dim=-1).mean().item()

        stride = 4 
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            curr_h_f = h_f[i:end]
            curr_h_l = h_l[i:end]
            curr_t_ids = target_ids[i:end]

            with torch.no_grad():
                h_scouts = curr_h_l[::stride] 

                scan_logits = torch.matmul(h_scouts, w_low.t())

                k_per_scout = n_candidates // (chunk_size // stride)
                _, top_indices = torch.topk(scan_logits, k_per_scout, dim=1)
                
                top_indices = top_indices.reshape(-1)

            w_f_pos = embedding_weight[curr_t_ids]
            w_f_cand = embedding_weight[top_indices]
            
            pos_sims = (curr_h_f * w_f_pos).sum(dim=-1, keepdim=True)
            neg_sims = torch.matmul(curr_h_f, w_f_cand.t())
            
            is_target = (top_indices.unsqueeze(0) == curr_t_ids.unsqueeze(1))
            neg_sims = neg_sims.masked_fill(is_target, float('-inf'))
            logits_m = torch.cat([pos_sims, neg_sims], dim=1).float()
            
            vocab_size = 262144
            V_rem = vocab_size - top_indices.size(0) - 1
            full_dim = hidden_states.size(1)
            
            # adding a ghost token as a sink to keep gradients hot
            curr_h_sq = (curr_h_f ** 2).sum(dim=-1, keepdim=True).float()
            var_m = (curr_h_sq * w_norm_sq) / full_dim
            ghost_logits_m = math.log(max(1, V_rem)) + (var_m / 2.0)
            
            logits_m = torch.cat([logits_m, ghost_logits_m], dim=1)
            
            # log softmax for stability
            log_p_m = F.log_softmax(logits_m, dim=-1)
            loss_m = -log_p_m[:, 0].sum()

            #   aux --------------------------------
            w_l_pos = w_f_pos[:, :low_rank_dim]
            w_l_cand = w_f_cand[:, :low_rank_dim]
            
            low_pos = (curr_h_l * w_l_pos).sum(dim=-1, keepdim=True)
            low_neg = torch.matmul(curr_h_l, w_l_cand.t())
            
            low_neg = low_neg.masked_fill(is_target, float('-inf'))
            logits_a = torch.cat([low_pos, low_neg], dim=1).float()
            
            curr_h_l_sq = (curr_h_l ** 2).sum(dim=-1, keepdim=True).float()
            var_a = (curr_h_l_sq * w_low_norm_sq) / low_rank_dim
            ghost_logits_a = math.log(max(1, V_rem)) + (var_a / 2.0)
            
            logits_a = torch.cat([logits_a, ghost_logits_a], dim=1)

            log_p_a = F.log_softmax(logits_a, dim=-1)
            loss_a = -log_p_a[:, 0].sum()

            total_loss += (loss_m + aux_weight * loss_a)

            # save softmax probs
            saved_chunks.append((
                curr_h_f, curr_t_ids, top_indices, 
                log_p_m.exp().to(dtype), log_p_a.exp().to(dtype), 
            ))

        ctx.save_for_backward(hidden_states, embedding_weight)
        ctx.aux_weight = aux_weight
        ctx.low_rank_dim = low_rank_dim
        ctx.saved_chunks = saved_chunks
        
        return total_loss / N

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, embedding_weight = ctx.saved_tensors
        aux_weight = ctx.aux_weight
        
        N, _ = hidden_states.shape
        total_tokens = N
        dtype = hidden_states.dtype
        
        grad_h = torch.zeros_like(hidden_states, dtype=torch.float32)
        grad_embed = torch.zeros_like(embedding_weight, dtype=torch.float32)

        chunk_size = ctx.saved_chunks[0][0].shape[0]
        grad_embed_low = grad_embed[:, :ctx.low_rank_dim]

        for i, chunk in enumerate(ctx.saved_chunks):
            (h_f, t_ids, top_idx, p_m, p_a) = chunk

            w_fp = embedding_weight[t_ids].to(dtype)
            w_fc = embedding_weight[top_idx].to(dtype)
            w_lp = w_fp[:, :ctx.low_rank_dim]
            w_lc = w_fc[:, :ctx.low_rank_dim]
            
            # index 0 = label
            # CE gradient = (softmax - label)
            dz_m = p_m.float().clone()
            dz_m[:, 0] -= 1.0
            dz_m = dz_m * (grad_output.float() / total_tokens)
            
            dz_a = p_a.float().clone()
            dz_a[:, 0] -= 1.0
            dz_a = dz_a * (grad_output.float() / total_tokens) * aux_weight

            dz_m_dt, dz_a_dt = dz_m.to(dtype), dz_a.to(dtype)

            gh = dz_m_dt[:, :1] * w_fp
            gh += torch.matmul(dz_m_dt[:, 1:-1], w_fc)
            ga = dz_a_dt[:, :1] * w_lp + torch.matmul(dz_a_dt[:, 1:-1], w_lc)
            
            gh[:, :ctx.low_rank_dim] += ga
            grad_h[i*chunk_size : i*chunk_size + h_f.shape[0]] = gh.float()

            h_f_float = h_f.float()
            grad_embed.index_add_(0, t_ids, dz_m[:, :1] * h_f_float)
            h_f = h_f.to(dtype)
            
            grad_embed.index_add_(0, top_idx, torch.matmul(dz_m_dt[:, 1:-1].t(), h_f).float())

            h_l_float = h_f_float[:, :ctx.low_rank_dim] 
            grad_embed_low.index_add_(0, t_ids, dz_a[:, :1] * h_l_float)
            grad_embed_low.index_add_(0, top_idx, torch.matmul(dz_a_dt[:, 1:-1].t(), h_f[:, :ctx.low_rank_dim]).float())
 
        return grad_h.to(dtype), grad_embed.to(dtype), None, None, None, None, None

class MatryoshkaSampledSoftmaxLoss(torch.nn.Module):
    def __init__(self, embedding_weight, low_rank_dim=64, n_candidates=2048, chunk_size=32):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.low_rank_dim = low_rank_dim
        self.n_candidates = n_candidates
        self.chunk_size = chunk_size

    def forward(self, hidden_states, target_ids):
        return MatryoshkaManualFunction.apply(
            hidden_states, 
            self.embedding_weight, 
            target_ids, 
            self.low_rank_dim,
            self.n_candidates,
            self.chunk_size,
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
    config.initial_context_length = MAX_LENGTH
    config.use_yarn = LONG_CONTEXT_TRAINING

    # for muons stability, we init the model to fp32
    model = Model(config, device).float()

    if LONG_CONTEXT_TRAINING or SKIP_BLOCK_MASK:
        blockswap_attention_layers(model)
        model.use_custom_ckpt_fn = True

    model.to(device)
    dataset = torch.utils.data.DataLoader(
        dataset, num_workers = 4, prefetch_factor = 2, batch_size=None, pin_memory = True
    )

    prefetch = CUDAPreFetch(dataset, device)
    prefetch.async_load()
    model.gradient_checkpointing = True


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
        model.load_state_dict(new_state_dict, strict=True)
        del new_state_dict, checkpoint
    
    model.float()
    
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
            model.layers[i] = torch.compile(block, mode="default", fullgraph=not (LONG_CONTEXT_TRAINING or SKIP_BLOCK_MASK))
    else:
        model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        output_device=local_rank,
        device_ids=[local_rank],
        find_unused_parameters=False,
        bucket_cap_mb=100,
        static_graph=False
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
                if Settings.SFT_TRAINING:
                    input_ids, labels = inputs
                else:
                    # pretraining + long context training
                    input_ids = inputs
                    labels = input_ids

                if not SKIP_BLOCK_MASK:
                    attention_mask, position_ids = get_packed_mask_and_pos_ids(input_ids, eos_token)
                else:
                    attention_mask, position_ids = None, None

                logits = model(
                    input_ids, attention_mask=attention_mask, cache_position=position_ids, return_hidden=True
                )

                eos_mask = (input_ids == eos_token)
                if Settings.SFT_TRAINING:
                    sft_mask = (labels[:, 1:] != -100).reshape(-1)
                    loss_mask = (~eos_mask[:, :-1].reshape(-1)) & sft_mask
                else:
                    loss_mask = ~eos_mask[:, :-1].reshape(-1)

                logits = logits[:, :-1, :].reshape(-1, model.module.config.hidden_size)[loss_mask]
                labels = labels[:, 1:].long().reshape(-1)[loss_mask]

                if USE_FAST_SOFTMAX:
                    loss = loss_fn(logits, labels)
                else:
                    loss = loss_fn(
                        model.module.lm_head.weight,
                        logits,
                        labels,
                    )

                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
        
        running_loss += loss.detach().item()
        if (step + 1) % ACCUM_STEPS == 0:
            if not Settings.use_adamw_only:
                scaler.unscale_(main_optimizer)
            scaler.unscale_(second_optimizer)

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

            if local_rank == 0 and ((step + 1) // ACCUM_STEPS) % 2 == 0:
                print(
                    f"step {(step // ACCUM_STEPS):05d} | loss {(smoothed_loss):.4f}"
                )
                losses.append(loss.detach().cpu())
            
            running_loss = 0
            

        del input_ids, attention_mask, logits, loss, labels

        batch = next_batch
        step += 1

        opt_step = (step + 1) // ACCUM_STEPS
        if (step + 1) % ACCUM_STEPS == 0 and local_rank == 0:
            if opt_step % SAVE_EVERY_STEP == 0:
                print(f"saving checkpoint at update step {opt_step}...")
                save_name = f"model_{opt_step}.safetensors"
                save_model(get_raw_model(model), save_name)
                update_model_hf(os.path.abspath(save_name), full_replace=True)

        if step == (TOTAL_NUMBER_OF_STEPS * ACCUM_STEPS):
            break

    dist.barrier()

    if local_rank == 0:
        save_model(get_raw_model(model), "model.safetensors")
        if hasattr(os, 'sync'):
            os.sync()
        update_model_hf(os.path.abspath("model.safetensors"), full_replace=True)
        print("model saved")
        plt.plot(losses)
        plt.savefig("loss_fig.png")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
