import json
import torch
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from model import Model, Config, RMSNorm
import torch.nn.functional as F
from lora import blockswap_attention_layers
from train import MatryoshkaSampledSoftmaxLoss
import torch.nn as nn

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
BATCH_SIZE = 2
DEVICE = "cuda"
torch.set_default_device(DEVICE)

REPO_ID = "yousefg/MaximusLLM"

config = Config.from_pretrained(REPO_ID)
config.hidden_size = 384
config.intermediate_size = 1024
config.num_hidden_layers = 4
config.num_attention_heads = 6
config.head_dim = 64
config.vocab_size = 262144

tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
iter_ds = iter(dataset)

def get_real_batch(target_len):
    batch_seqs = []
    while len(batch_seqs) < BATCH_SIZE:
        try:
            txt = next(iter_ds)["text"]
            toks = tokenizer(txt, truncation=True, max_length=target_len, return_tensors="pt")["input_ids"]
            if toks.size(1) >= target_len * 0.8:
                batch_seqs.append(toks)
        except StopIteration:
            return None
    
    batch = torch.full((BATCH_SIZE, target_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, t in enumerate(batch_seqs):
        curr_len = min(t.size(1), target_len)
        batch[i, :curr_len] = t[0, :curr_len]
    return batch.to(DEVICE)

def setup_standard(model):
    for layer in model.layers:
        self = layer.self_attn
        self.is_latent = False
        self.scaling = config.query_pre_attn_scalar ** -0.5
    
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, device=DEVICE
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device=DEVICE
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device=DEVICE
        )
    
        self.k_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps, device=DEVICE)
        self.q_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps, device=DEVICE)

def setup_sliding_window(model, window_size=1024):
    setup_standard(model)
    old_forward = model.forward

    def swa_mask_forward(input_ids, **kwargs):
        T = input_ids.size(1)
        mask = torch.zeros((T, T), dtype=torch.bool, device=input_ids.device)
        
        for i in range(T):
            start = max(0, i - window_size + 1) 
            mask[i, start : i + 1] = True
        
        kwargs['attention_mask'] = mask.unsqueeze(0).unsqueeze(0)
        return old_forward(input_ids, **kwargs)

    model.forward = swa_mask_forward

def setup_randnla(model):
    blockswap_attention_layers(model)

def run_ablation(name, setup_fn, train_steps=1000):
    print(f"\n{'='*40}\nRUNNING: {name}\n{'='*40}")
    results = {"vram": [], "speed": [], "ppl":[]}

    for length in SEQ_LENS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = Model(config, DEVICE, enabled_lm_head=True).to(torch.float32)
        setup_fn(model)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_fn = MatryoshkaSampledSoftmaxLoss(
            model.embed_tokens.weight, low_rank_dim=64, n_candidates=2048, chunk_size=32
        )

        print(f"--- Micro-Training {name} at Length {length} ---")
        try:
            for _ in range(train_steps):
                batch = get_real_batch(length)
                if batch is None: 
                    break
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = model(batch, attention_mask = None, return_hidden = True)
                    h_shift = hidden[:, :-1, :].reshape(-1, config.hidden_size)
                    t_shift = batch[:, 1:].reshape(-1)
                    loss = train_fn(h_shift, t_shift)
                    
                loss.backward()
                optimizer.step()
                
            model.eval()
            eval_batch = get_real_batch(length)
            
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
                    
                    start_event.record()
                    logits = model(eval_batch, attention_mask=None, return_hidden=False)

                    end_event.record()
                    torch.cuda.synchronize()
                    dur = start_event.elapsed_time(end_event) / 1000.0

                    shift_logits = logits[:, :-1, :].contiguous().view(-1, config.vocab_size)
                    shift_labels = eval_batch[:, 1:].contiguous().view(-1)
                    val_loss = F.cross_entropy(shift_logits, shift_labels)
                    ppl = torch.exp(val_loss).item()

            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

            print(f"Len {length:5} | PPL: {ppl:7.2f} | VRAM: {peak_vram:5.2f}GB | Speed: {length/dur:7.1f} tok/s")
            
            results["ppl"].append(ppl)
            results["vram"].append(peak_vram)
            results["speed"].append(dur)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Len {length:5} | CRASHED: OOM")
                results["ppl"].append(float('nan'))
                results["vram"].append(float('nan'))
                results["speed"].append(float('nan'))
            else: 
                raise e
        
        del model, optimizer, batch, eval_batch

    return results

res_std = run_ablation("GQA (Quadratic)", setup_standard)
res_lin = run_ablation("Linear (Pure Compression)", setup_sliding_window)
res_rnd = run_ablation("RandNLA (Bifurcated)", setup_randnla)

def plot_ablation():
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    def to_loss(ppl_list):
        return [math.log(p) if (p > 0 and p != float('inf')) else float('nan') for p in ppl_list]

    ax1.plot(SEQ_LENS, to_loss(res_std["ppl"]), label='GQA (Standard)', marker='o', color='black', linewidth=1.5)
    ax1.plot(SEQ_LENS, to_loss(res_lin["ppl"]), label='Linear (SWA)', marker='x', linestyle='--', color='red', linewidth=1.5)
    ax1.plot(SEQ_LENS, to_loss(res_rnd["ppl"]), label='RandNLA (Maximus)', marker='s', color='blue', linewidth=2.5)
    
    ax1.set_title("Semantic Stability (Val Loss)\n(Lower is Better)")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.plot(SEQ_LENS, res_std["vram"], label='Standard', marker='o', color='black')
    ax2.plot(SEQ_LENS, res_rnd["vram"], label='RandNLA', marker='s', color='blue', linewidth=2.5)
    if any(not math.isnan(x) for x in res_lin["vram"]):
        ax2.plot(SEQ_LENS, res_lin["vram"], label='Linear', marker='x', linestyle='--', color='red')

    ax2.set_title("Memory Usage\n(Lower is Better)")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak VRAM (GB)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    def calc_throughput(results):
        limit = len(results["speed"])
        return [L/T for L, T in zip(SEQ_LENS[:limit], results["speed"])]

    ax3.plot(SEQ_LENS[:len(res_std["speed"])], calc_throughput(res_std), label='Standard', marker='o', color='black')
    ax3.plot(SEQ_LENS[:len(res_lin["speed"])], calc_throughput(res_lin), label='Linear', marker='x', linestyle='--', color='red')
    ax3.plot(SEQ_LENS[:len(res_rnd["speed"])], calc_throughput(res_rnd), label='RandNLA', marker='s', color='blue', linewidth=2.5)
    
    ax3.set_title("Throughput\n(Higher is Better)")
    ax3.set_xlabel("Sequence Length")
    ax3.set_ylabel("Tokens per Second")
    ax3.legend()
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('randnla_ablation_full.png', dpi=300)
    print("\nFull Ablation Study Plot saved: randnla_ablation_full.png")

plot_ablation()

results = {
    "standard_attention": {
        "perplexity": res_std["ppl"],
        "vram": res_std["vram"],
        "speed": res_std["speed"],
    },
    "linear_attention": {
        "perplexity": res_lin["ppl"],
        "vram": res_lin["vram"],
        "speed": res_lin["speed"],
    },
    "randnla_attention": {
        "perplexity": res_rnd["ppl"],
        "vram": res_rnd["vram"],
        "speed": res_rnd["speed"],
    },
}

with open("loss_results.json", "w") as f:
    json.dump(results, f, indent=4)
