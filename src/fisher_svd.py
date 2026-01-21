import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

token = "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"
model_id = "google/gemma-3-270m"

NUM_CALIBRATION_BATCHES = 16 
BATCH_SIZE = 4
SEQ_LEN = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# get the importance of the parameters and how they affect the input and gradients
def compute_fisher_importance(model, num_batches = 100):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True
    )
    
    fisher_info = {}
    
    model.train()
    model.to(device)
    for i, batch in tqdm(enumerate(dataset), total=num_batches):
        if i >= num_batches: 
            break
        
        text = batch['text']
        if len(text) < 10:
            continue 
        
        inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        
        logits = model(input_ids.to(device), attention_mask = torch.ones_like(input_ids), output_hidden_states=True).logits
        if logits.ndim != 3:
            logits = logits.unsqueeze(0)
        if input_ids.ndim != 2:
            input_ids = input_ids.unsqueeze(0)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        input_ids = input_ids[:, 1:].reshape(-1)

        loss = torch.nn.functional.cross_entropy(logits, input_ids)
        
        model.zero_grad()
        loss.backward()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    # sum((dL/dW)^2)
                    g_sq = module.weight.grad.detach().pow(2).sum(dim=0)
                    
                    if name not in fisher_info:
                        fisher_info[name] = g_sq
                    else:
                        fisher_info[name] += g_sq

    # normalization
    for name in fisher_info:
        avg_grad = fisher_info[name] / num_batches
        fisher_info[name] = torch.sqrt(avg_grad)
        fisher_info[name] += 1e-6 

    return fisher_info

# decompose a pretrained normal Attention layer -> a latent attention layer
def svd_init_latent(layer_a, layer_b, orig_weights, rank, norm_layer, orig_norm, fisher_vector, debug_information = True):

    gemma = orig_norm.weight.data.float()#.repeat(num_heads)
    W = orig_weights.weight.float()

    num_repeats = W.size(0) // gemma.size(0) # num_heads
    gemma = gemma.repeat(num_repeats)

    W_fused = W * gemma.unsqueeze(1)
    
    f_max = fisher_vector.max()
    fisher_scale = torch.clamp(fisher_vector / (f_max + 1e-8), min = 0.01).unsqueeze(0)

    W = W_fused * fisher_scale

    U, S, V_h = torch.linalg.svd(W, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = V_h[:rank, :]

    Vh_r /= fisher_scale

    #scale
    S_sqrt = torch.diag(torch.sqrt(S_r))

    W_a = torch.matmul(S_sqrt, Vh_r)
    W_b = torch.matmul(U_r, S_sqrt)

    layer_a.weight.data.copy_(W_a)
    layer_b.weight.data.copy_(W_b)

    torch.nn.init.ones_(norm_layer.weight)

    # reconstruction error
    if debug_information:
        W_recon = torch.matmul(W_b, W_a)
        diff = torch.norm(W_fused - W_recon) / torch.norm(W_fused)
        
        w_err = torch.norm((W_fused - W_recon) * fisher_scale) / torch.norm(W_fused * fisher_scale)
        
        print(f"  Reconstruction Error: {diff:.4f} | Weighted Error: {w_err:.4f}")

