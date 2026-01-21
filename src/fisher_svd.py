import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

token = ""
model_id = "google/gemma-3-270m"

NUM_CALIBRATION_BATCHES = 16 
BATCH_SIZE = 4
SEQ_LEN = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

# get the importance of the parameters and how they affect the input and gradients
def compute_fisher_importance(model, num_batches = 100):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True
    )
    
    fisher_info = {}
    
    model.train()
    for i, batch in tqdm(enumerate(dataset), total=num_batches):
        if i >= num_batches: 
            break
        
        text = batch['text']
        if len(text) < 10:
            continue 
        
        inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
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

    W = orig_weights.weight * gemma.unsqueeze(0)
    W = W * fisher_vector.unsqueeze(0)

    U, S, V_h = torch.linalg.svd(W, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = V_h[:rank, :]

    Vh_r /= fisher_vector

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
        diff = torch.norm(W - W_recon) / torch.norm(W)
        print(f"Reconstruction Error: {diff}")

