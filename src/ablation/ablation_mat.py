import json
import torch
import random
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from model import Model, Config
from train import MatryoshkaSampledSoftmaxLoss

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCALING_CONFIGS = [
    {"name": "180M", "h": 512,  "layers": 8,  "heads": 8},
    {"name": "440M", "h": 1024, "layers": 12, "heads": 16},
    {"name": "1.2B", "h": 2048, "layers": 14, "heads": 32},
]

SEEDS = [42, 123, 2035]

repo_id = "yousefg/MaximusLLM"
config = Config.from_pretrained(repo_id)

SAMPLES_TO_TRAIN = 4000
BATCH_SIZE = 8
SEQ_LEN = 512
DEVICE = "cuda"
EVAL_INTERVAL = 500

def set_seed(seed: int):
    if seed is None:
        print("SEED IS NONE!", flush=True)
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Fixed seed set to: {seed}")

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
            if toks.size(1) >= SEQ_LEN * 0.7:
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


# for validation we will run cross entropy for both of them to ensure fairness
def run_experiment(name, loss_type, seed, scale = 1.0):
    set_seed(seed)

    print("\n======================================")
    print(f"STARTING RUN: {name}")
    print("======================================")
    model = Model(config, DEVICE)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            std = 0.02
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    model.apply(init_weights)

    if scale != 1.0:
        scale *= 1000
    train_buffer = [get_next_batch() for _ in range((SAMPLES_TO_TRAIN + scale) // BATCH_SIZE)]

    print(f">>> Loaded {len(train_buffer)} Training Batches.")


    model.embed_tokens.embed_scale = torch.tensor(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.to(DEVICE)
    model.gradient_checkpointing = True

    if loss_type == "liger":
        if LigerFusedLinearCrossEntropyLoss:
            train_fn = LigerFusedLinearCrossEntropyLoss()
        else:
            train_fn = nn.CrossEntropyLoss()
    else:
        train_fn = MatryoshkaSampledSoftmaxLoss(
            model.embed_tokens.weight, low_rank_dim=128,
            n_candidates=min(8192, model.config.vocab_size // 64), chunk_size=max(128, model.config.hidden_size // 8)
        )

    val_fn = nn.CrossEntropyLoss()
    
    steps = []
    times = []
    val_losses = []
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    accumulated_loss_time = 0.0
    peak_vram = 0.0
    for step, inputs in enumerate(train_buffer):
        inputs = inputs.to(device=DEVICE, dtype=torch.long, non_blocking=True)

        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            hidden = model(inputs, attention_mask=None, return_hidden=True)
            h_shift = hidden[:, :-1, :].reshape(-1, config.hidden_size)
            t_shift = inputs[:, 1:].reshape(-1)

            if step == 2:
                torch.cuda.reset_peak_memory_stats()
        
            h_bench = h_shift.detach().requires_grad_(True)
            torch.cuda.synchronize()
            start_event.record()
            if loss_type == "liger":
                loss = train_fn(model.embed_tokens.weight, h_bench, t_shift)
            else:
                loss = train_fn(h_bench, t_shift)
            
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()

            if step == 2:
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            # plug the previous gradients to save compute
            h_shift.backward(gradient=h_bench.grad)

        optimizer.step()
        accumulated_loss_time += start_event.elapsed_time(end_event) / 1000.0
        
        if step % EVAL_INTERVAL == 0 or step == len(train_buffer) - 1:
            del hidden, h_shift
            del inputs
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for v_in in val_buffer:
                    v_in = v_in.to(DEVICE, dtype=torch.long, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                        v_h = model(v_in, attention_mask=None, return_hidden=True)
                        logits = torch.matmul(v_h, model.embed_tokens.weight.t())
                    
                        l_shift = logits[:, :-1, :].reshape(-1, config.vocab_size)
                        vt_shift = v_in[:, 1:].reshape(-1)
                    del v_in
                    del v_h
                    v_loss = val_fn(l_shift, vt_shift)
                    total_val_loss += v_loss.item()
            
            avg_val_loss = total_val_loss / len(val_buffer)
            print(f"Step {step} | Train Time: {accumulated_loss_time:.1f}s | Val Loss: {avg_val_loss:.4f}")
            
            steps.append(step)
            times.append(accumulated_loss_time)
            val_losses.append(avg_val_loss)

    avg_speed = len(train_buffer) / accumulated_loss_time
    
    print(f"\n>>> FINAL STATS FOR {name}:")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Speed:     {avg_speed:.2f} steps/sec")

    del model
    del optimizer
    del train_fn
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return steps, times, val_losses, peak_vram, avg_speed

all_results = {}
for i, conf in enumerate(SCALING_CONFIGS):
    config.hidden_size = conf["h"]
    config.num_hidden_layers = conf["layers"]
    config.num_attention_heads = conf["heads"]
    config.intermediate_size = conf["h"] * 4

    should_break = False
    try:
        sl, tl, vl, vraml, speedl = run_experiment(f"{conf['name']}-Liger", "liger", seed=SEEDS[i], scale = i + 1)
        sm, tm, vm, vramm, speedm = run_experiment(f"{conf['name']}-MAXIS", "maxis", seed=SEEDS[i], scale = i + 1)
    except Exception as e:
        print(e, flush=True)
        pass
    
    all_results[conf["name"]] = {
        "liger": {"steps": sl, "time": tl, "val": vl, "vram": vraml, "speed": speedl},
        "maxis": {"steps": sm, "time": tm, "val": vm, "vram": vramm, "speed": speedm}
    }
    if should_break:
        break

def plot_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    steps_l, time_l, val_l, vram_l, speed_l = all_results["1.2B"]["liger"]
    steps_m, time_m, val_m, vram_m, speed_m = all_results["1.2B"]["maxis"]
    # due to embedding scaling inherient in the model
    target_start_loss = 12.5 
    scaling_factor = val_l[0] / target_start_loss
    
    norm_val_l = [x / scaling_factor for x in val_l]
    norm_val_m = [x / scaling_factor for x in val_m]

    ax1.plot(time_l, norm_val_l, label='Standard CE (Liger)', marker='o', color='gray', linestyle='-', linewidth=1.5)
    ax1.plot(time_m, norm_val_m, label='MAXIS (Ours)', marker='s', color='blue', linewidth=2.5)
    
    ax1.set_title("Intelligence per Second\n(Lower & Left is Better)")
    ax1.set_xlabel("Training Time (Seconds)")
    ax1.set_ylabel("Validation Loss")
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

with open("loss_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

plot_results()
