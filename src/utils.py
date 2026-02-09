from huggingface_hub import upload_file, delete_file
from safetensors.torch import save_file
import torch.distributed as dist
import torch.nn.functional as F
import tempfile
import torch
import os

def update_model_hf(model_path, hf_dir="yousefg/MaximusLLM", token="", full_replace=False):
    with tempfile.TemporaryDirectory() as temp_dir:
        
        file_to_upload = model_path
        
        if model_path.endswith(".pt"):            
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            for k, v in state_dict.items():
                state_dict[k] = v.to(torch.float16)

            shared_keys = ['module._orig_mod.lm_head.weight', 'module._orig_mod.embed_tokens.weight']

            for key in shared_keys:
                if key in state_dict:
                    state_dict[key] = state_dict[key].clone().contiguous()

            temp_safe_path = os.path.join(temp_dir, "model.safetensors")
            save_file(state_dict, temp_safe_path)
            
            file_to_upload = temp_safe_path

        upload_file(
            path_or_fileobj=file_to_upload,
            path_in_repo="model_test.safetensors" if not full_replace else "model.safetensors",
            repo_id=hf_dir,
            token=token
        )

        # will uncomment
        #if full_replace:
        #    delete_file(
        #        path_in_repo="model_test.safetensors",
        #        repo_id = hf_dir,
        #        token=token
        #    )

def get_raw_model(model):
    def clean(m):
        if hasattr(m, "module"):
            m = m.module
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m
    model = clean(model)
    for i, layer in enumerate(model.layers):
        model.layers[i] = clean(layer)
    return model

def clean_checkpoint(checkpoint):
    new_state_dict = {}

    for k, v in checkpoint.items():
        new_k = k.replace("module.", "")
        new_k = new_k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
        
    return new_state_dict

def get_global_loss(running_loss, world_size):
    if not (world_size > 1):
        return running_loss
    t = torch.tensor([running_loss], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size

def print_header(title):
    print(f"\n{'='*60}\n {title} \n{'='*60}")

def analyze_model(model, tokenizer, config):

    DEVICE = "cuda"
    model.eval()

    print_header("TEST 1: EMBEDDING SPACE HEALTH")
    
    W = model.embed_tokens.weight.data
    norms = W.norm(dim=-1)
    print(f"Embedding Norms -> Mean: {norms.mean().item():.4f} | Std: {norms.std().item():.4f}")
    
    indices = torch.randint(0, config.vocab_size, (1000,), device=DEVICE)
    subset = W[indices]
    subset_norm = F.normalize(subset, p=2, dim=-1)
    
    sim_matrix = torch.matmul(subset_norm, subset_norm.t())

    sim_matrix.fill_diagonal_(0)
    avg_sim = sim_matrix.sum() / (1000 * 999)
    
    print(f"Average Cosine Similarity between random words: {avg_sim.item():.4f}")
    
    if avg_sim > 0.9:
        print("CRITICAL: Vector Collapse. All words mean the same thing!")
    elif avg_sim < 0.001:
        print("CRITICAL: White Noise. Embeddings are random!")
    else:
        print("Embeddings look healthy (distinct but related)!")

    print_header("TEST 2: MATRYOSHKA HIERARCHY")
    
    full_sim = F.cosine_similarity(subset_norm[0], subset_norm[1], dim=0)
    
    low_dim = 64
    subset_low = F.normalize(subset[:, :low_dim], p=2, dim=-1)
    low_sim = F.cosine_similarity(subset_low[0], subset_low[1], dim=0)
    
    print(f"Sim(Word A, Word B) Full-Rank: {full_sim.item():.4f}")
    print(f"Sim(Word A, Word B) Low-Rank:  {low_sim.item():.4f}")
    
    if abs(full_sim - low_sim) > 0.5:
        print("Warning: Low-rank representation disagrees strongly with Full-rank!")
    else:
        print("Matryoshka structure detected")

    print_header("TEST 3: LAYER ACTIVATIONS")
    
    input_text = "The capital of France is "
    input_ids = torch.tensor(tokenizer(input_text)["input_ids"], device=DEVICE).unsqueeze(0)
    
    # hook to capture activations
    activations = {}
    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            activations[name] = t.detach()
        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(get_hook(f"Layer_{i}")))

    with torch.no_grad():
        final_out = model(input_ids, attention_mask=None, return_hidden=True)
        if isinstance(final_out, tuple):
            final_out = final_out[0]

    print(f"{'Layer':<10} | {'Mean':<10} | {'Std':<10} | {'Max':<10}")
    for i in range(len(model.layers)):
        act = activations[f"Layer_{i}"]
        print(f"{i:<10} | {act.mean().item():.4f}     | {act.std().item():.4f}     | {act.max().item():.4f}")
        
        if act.isnan().any():
            print(f"LAYER {i} HAS NaN VALUES!")

    final_norm = final_out.norm(dim=-1).mean().item()
    print(f"Final RMSNorm Output Magnitude: {final_norm:.4f}")
    
    if final_norm < 0.1:
        print("CRITICAL: Signal vanished. The Brain died!")
    elif final_norm > 1000:
        print("CRITICAL: Signal exploded. The Brain fried!")

    print_header("TEST 4: RoPE GEOMETRY CHECK")
    
    hidden = model.embed_tokens(input_ids)
    
    seq_len = input_ids.shape[1]
    pos_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    cos, _ = model.rotary_emb._compute_cos_sin(seq_len)
    
    try:
        t_cos = cos[pos_ids].unsqueeze(2)
        print(f"RoPE Shape check: {t_cos.shape} vs Hidden {hidden.shape}")
        if t_cos.shape[1] != hidden.shape[1]:
             print("RoPE Sequence Length mismatch!")
        else:
             print("RoPE shapes align.")
    except Exception as e:
        print(f"RoPE Logic Crash: {e}")

    print_header("TEST 5: PREDICTION LOGIC")
    
    last_token_hidden = final_out[:, -1, :]
    
    h_norm = F.normalize(last_token_hidden, p=2, dim=-1)
    w_norm = F.normalize(model.embed_tokens.weight, p=2, dim=-1)
    
    for test_scale in [1.0, 20.0, 50.0, 94]:
        logits = torch.matmul(h_norm, w_norm.t()) * test_scale
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(probs, 3)
        
        print(f"\n--- Scale {test_scale} ---")
        for i in range(3):
            token_str = tokenizer.decode([top_idx[0, i].item()])
            print(f"Pred {i+1}: '{token_str}' ({top_val[0, i].item():.4f})")

    for h in hooks: 
        h.remove()
