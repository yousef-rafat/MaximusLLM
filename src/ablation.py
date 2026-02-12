from copy import deepcopy
from model import Model, Config
from train import MatryoshkaSampledSoftmaxLoss
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from fisher_svd import svd_init_latent, compute_fisher_importance # TODO
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
import torch


repo_id = "yousefg/MaximusLLM"
config = Config.from_pretrained(repo_id)
model_base = Model(config, "cuda")
model_maxis = deepcopy(model_base)
STEPS = 1000
BATCH_SIZE = 8
SEQ_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(repo_id)
dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)

data_buffer = []
iter_ds = iter(dataset)

for _ in range(STEPS):
    tokens_list = []
    while len(tokens_list) < BATCH_SIZE:
        try:
            text = next(iter_ds)["text"]
            enc = tokenizer(text, truncation=True, max_length=SEQ_LEN, return_tensors="pt")["input_ids"]
            if enc.size(1) > 100:
                tokens_list.append(enc)
        except StopIteration:
            break
            
    max_len = max([t.size(1) for t in tokens_list])
    padded_batch = torch.full((BATCH_SIZE, max_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded_batch[i, :t.size(1)] = t
        
    data_buffer.append(padded_batch.pin_memory())

print(f">>> Loaded {len(data_buffer)} batches. Starting race.")

def run_ablation(model_name, model, loss_type):
    print(f"\n>>> Starting Run: {model_name}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    if loss_type == "liger":
        criterion = LigerFusedLinearCrossEntropyLoss()
    elif loss_type == "maxis":
        criterion = MatryoshkaSampledSoftmaxLoss(
            model.embed_tokens.weight, 
            low_rank_dim=64, 
            n_candidates=2048, 
            chunk_size=32
        )
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    history = []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    
    for i in range(5):
        inputs = data_buffer[i]
        optimizer.zero_grad()
        out = model(inputs, return_hidden=True)
        loss = out.sum() * 0.0 + 1.0
        loss.backward()
        optimizer.step()

    total_gpu_time = 0.0
    
    for step in range(STEPS):
        inputs = data_buffer[step]
        
        optimizer.zero_grad()
        
        start_event.record()
        
        hidden = model(inputs, return_hidden=True)
        
        h_shifted = hidden[:, :-1, :].reshape(-1, config.hidden_size)
        t_shifted = inputs[:, 1:].reshape(-1)
        
        if loss_type == "liger":
            loss = criterion(model.embed_tokens.weight, h_shifted, t_shifted)
        elif loss_type == "maxis":
            loss = criterion(h_shifted, t_shifted)

        loss.backward()
        optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        step_ms = start_event.elapsed_time(end_event)
        total_gpu_time += step_ms
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.4f} | Time: {step_ms:.2f}ms")
            
        history.append({
            "step": step,
            "loss": loss.item(),
            "time_ms": step_ms
        })

    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
    avg_time = total_gpu_time / STEPS / 1000.0
    
    print(f"Finished {model_name}.")
    print(f"Avg Time/Step: {avg_time:.4f}s")
    print(f"Peak VRAM:     {max_mem:.2f} GB")
    
    return history, max_mem, avg_time

def run_mat_ablation(model_base, model_maxis):
    hist_liger, mem_liger, time_liger = run_ablation("Liger (Standard CE)", model_base, "liger")

    del model_base
    torch.cuda.empty_cache()

    hist_maxis, mem_maxis, time_maxis = run_ablation("MAXIS (Matryoshka)", model_maxis, "maxis")

    def plot_results():
        l_loss = [x['loss'] for x in hist_liger]
        m_loss = [x['loss'] for x in hist_maxis]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(l_loss, label='Standard CE (Liger)', alpha=0.7)
        plt.plot(m_loss, label='MAXIS (Yours)', linewidth=2)
        plt.title('Training Convergence')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        metrics = ['Peak VRAM (GB)', 'Time per 100 Steps (s)']
        liger_vals = [mem_liger, time_liger]
        maxis_vals = [mem_maxis, time_maxis]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], liger_vals, width, label='Standard CE')
        plt.bar([i + width/2 for i in x], maxis_vals, width, label='MAXIS')
        plt.xticks(x, metrics)
        plt.title('Efficiency Comparison')
        plt.legend()
        
        plt.savefig('ablation_study.png')
        print("Plot saved to ablation_study.png")
        
        print("\n=== FINAL RESULTS TABLE ===")
        print(f"{'Metric':<20} | {'Standard CE':<15} | {'MAXIS':<15} | {'Improvement'}")
        print("-" * 65)
        print(f"{'Peak VRAM':<20} | {mem_liger:.2f} GB        | {mem_maxis:.2f} GB        | {((mem_liger-mem_maxis)/mem_liger)*100:.1f}% Less")
        print(f"{'Total Time':<20} | {time_liger:.1f} s         | {time_maxis:.1f} s         | {((time_liger-time_maxis)/time_liger)*100:.1f}% Faster")

    plot_results()