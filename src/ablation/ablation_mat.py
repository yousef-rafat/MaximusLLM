import json
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from model import Model, Config
from train import MatryoshkaSampledSoftmaxLoss

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

# configure to be 40m approx.
repo_id = "yousefg/MaximusLLM"
config = Config.from_pretrained(repo_id)

config.hidden_size = 384
config.intermediate_size = 1024
config.num_hidden_layers = 6
config.num_attention_heads = 6
config.head_dim = 64
config.vocab_size = 262144

SAMPLES_TO_TRAIN = 4000
BATCH_SIZE = 8
SEQ_LEN = 512 
DEVICE = "cuda"
EVAL_INTERVAL = 25

torch.set_default_dtype(torch.float32)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

print(">>> Pre-loading data into RAM...")
data_buffer = []
iter_ds = iter(dataset)

def get_next_batch():
    batch_seqs = []
    while len(batch_seqs) < BATCH_SIZE:
        try:
            txt = next(iter_ds)["text"]
            toks = tokenizer(txt, truncation=True, max_length=SEQ_LEN, return_tensors="pt")["input_ids"]
            if toks.size(1) > 50:
                batch_seqs.append(toks)
        except StopIteration: 
            return None
    
    max_l = max([t.size(1) for t in batch_seqs])
    batch = torch.full((BATCH_SIZE, max_l), tokenizer.pad_token_id, dtype=torch.long)
    for i, t in enumerate(batch_seqs):
        batch[i, :t.size(1)] = t
    return batch.pin_memory()

if "val_buffer" not in globals():
    val_buffer = [get_next_batch() for _ in range(20)]
    train_buffer = [get_next_batch() for _ in range(SAMPLES_TO_TRAIN // BATCH_SIZE)]

print(f">>> Loaded {len(train_buffer)} Training Batches.")


# for validation we will run cross entropy for both of them to ensure fairness
def run_experiment(name, loss_type):
    print("\n======================================")
    print(f"STARTING RUN: {name}")
    print("======================================")
    
    model = Model(config, DEVICE).to(torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.to(DEVICE)
    
    if loss_type == "liger":
        if LigerFusedLinearCrossEntropyLoss:
            train_fn = LigerFusedLinearCrossEntropyLoss()
        else:
            train_fn = nn.CrossEntropyLoss()
    else:
        train_fn = MatryoshkaSampledSoftmaxLoss(
            model.embed_tokens.weight, low_rank_dim=64, n_candidates=2048, chunk_size=32
        )

    val_fn = nn.CrossEntropyLoss()
    
    steps = []
    times = []
    val_losses = []
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    accumulated_train_time = 0.0
    
    for step, inputs in enumerate(train_buffer):
        inputs = inputs.to(device=DEVICE, dtype=torch.long, non_blocking=True)
        
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            hidden = model(inputs, attention_mask=None, return_hidden=True)
            h_shift = hidden[:, :-1, :].reshape(-1, config.hidden_size)
            t_shift = inputs[:, 1:].reshape(-1)
        
            if loss_type == "liger":
                loss = train_fn(model.embed_tokens.weight, h_shift, t_shift)
            else:
                loss = train_fn(h_shift, t_shift)
            
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        accumulated_train_time += (time.time() - t0)
        
        if step % EVAL_INTERVAL == 0 or step == len(train_buffer) - 1:
            del hidden, h_shift
            del inputs
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for v_in in val_buffer:
                    v_in = v_in.to(DEVICE, dtype=torch.long, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                        v_h = model(v_in, attention_mask=None, return_hidden=True)
                        logits = torch.matmul(v_h, model.embed_tokens.weight.t())
                    
                        l_shift = logits[:, :-1, :].reshape(-1, config.vocab_size)
                        vt_shift = v_in[:, 1:].reshape(-1)
                    del v_in
                    del v_h
                    v_loss = val_fn(l_shift, vt_shift)
                    total_val_loss += v_loss.item()
            
            avg_val_loss = total_val_loss / len(val_buffer)
            print(f"Step {step} | Train Time: {accumulated_train_time:.1f}s | Val Loss: {avg_val_loss:.4f}")
            
            steps.append(step)
            times.append(accumulated_train_time)
            val_losses.append(avg_val_loss)

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    avg_speed = len(train_buffer) / accumulated_train_time
    
    print(f"\n>>> FINAL STATS FOR {name}:")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Speed:     {avg_speed:.2f} steps/sec")
    
    return steps, times, val_losses, peak_vram, avg_speed

steps_l, time_l, val_l, vram_l, speed_l = run_experiment("Liger (Standard)", "liger")
steps_m, time_m, val_m, vram_m, speed_m = run_experiment("MAXIS (Yours)", "maxis")

def plot_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(time_l, val_l, label='Standard CE (Liger)', marker='o', color='gray')
    ax1.plot(time_m, val_m, label='MAXIS (Ours)', marker='s', color='blue', linewidth=2)
    ax1.set_title("Intelligence per Second\n(Lower & Left is Better)")
    ax1.set_xlabel("Training Time (Seconds)")
    ax1.set_ylabel("Validation Loss (True CE)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    metrics = ['Peak VRAM (GB)', 'Steps per Second']
    liger_vals = [vram_l, speed_l]
    maxis_vals = [vram_m, speed_m]
    
    x = range(len(metrics))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], liger_vals, width, label='Standard CE', color='gray')
    ax2.bar([i + width/2 for i in x], maxis_vals, width, label='MAXIS', color='blue')
    ax2.set_xticks(x, metrics)
    ax2.set_title("Hardware Efficiency")
    ax2.legend()
    
    for i, v in enumerate(liger_vals):
        ax2.text(i - width/2, v, f"{v:.2f}", ha='center', va='bottom')
    for i, v in enumerate(maxis_vals):
        ax2.text(i + width/2, v, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('maxis_benchmark.png', dpi=300)
    print("\nBenchmark plot saved to maxis_benchmark.png")
    
    vram_save = ((vram_l - vram_m) / vram_l) * 100
    speed_boost = ((speed_m - speed_l) / speed_l) * 100
    print(f"MAXIS saves {vram_save:.1f}% VRAM")
    print(f"MAXIS runs {speed_boost:.1f}% Faster")

plot_results()

results = {
    "cross_entropy": {
        "steps": steps_l,
        "time": time_l,
        "val": val_l,
        "vram": vram_l,
        "speed": speed_l
    },
    "maxis": {
        "steps": steps_m,
        "time": time_m,
        "val": val_m,
        "vram": vram_m,
        "speed": speed_m
    }
}

with open("loss_results.json", "w") as f:
    json.dump(results, f, indent=4)
