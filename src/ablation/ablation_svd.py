import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
from model import Model, Config
from fisher_svd import compute_fisher_importance, token, model_id

model = AutoModelForCausalLM.from_pretrained(
    model_id, token=token, torch_dtype=torch.float32, trust_remote_code=True
)

def svd_init_latent(layer_a, layer_b, orig_weights, rank, norm_layer, orig_norm, fisher_vector=None, use_fisher=True, debug_information=True):

    gemma = orig_norm.weight.data.float()
    W = orig_weights.weight.float()

    num_repeats = W.size(0) // gemma.size(0)
    gemma = gemma.repeat(num_repeats)

    W_fused = W * gemma.unsqueeze(1)
    
    if fisher_vector is not None:
        f_max = fisher_vector.max()
        fisher_scale = torch.clamp(fisher_vector / (f_max + 1e-8), min=0.01).unsqueeze(0)
    
    if use_fisher and fisher_vector is not None:
        W_svd_input = W_fused * fisher_scale
    else:
        W_svd_input = W_fused

    U, S, V_h = torch.linalg.svd(W_svd_input, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = V_h[:rank, :]

    if use_fisher and fisher_vector is not None:
        Vh_r /= fisher_scale

    S_sqrt = torch.diag(torch.sqrt(S_r))

    W_a = torch.matmul(S_sqrt, Vh_r)
    W_b = torch.matmul(U_r, S_sqrt)

    layer_a.weight.data.copy_(W_a)
    layer_b.weight.data.copy_(W_b)

    torch.nn.init.ones_(norm_layer.weight)

    if debug_information:
        W_recon = torch.matmul(W_b, W_a)
        
        diff = torch.norm(W_fused - W_recon) / torch.norm(W_fused)
        
        if fisher_vector is not None:
            w_err = torch.norm((W_fused - W_recon) * fisher_scale) / torch.norm(W_fused * fisher_scale)
        else:
            w_err = diff
        
        return diff.item(), w_err.item()

def test_fisher(model_test, fisher_vector, use_fisher=True):
    stats = {'q_diff': [], 'q_w_err': [], 'kv_diff':[], 'kv_w_err':[]}
    
    for i, module in enumerate(model_test.layers):
        if hasattr(module.self_attn, "q_a"):
            
            fisher_vector_q = fisher_vector.get(f"model.layers.{i}.self_attn.q_proj")
            fisher_vector_k = fisher_vector.get(f"model.layers.{i}.self_attn.k_proj")
            fisher_vector_v = fisher_vector.get(f"model.layers.{i}.self_attn.v_proj")
            fisher_vector_kv = fisher_vector_k + fisher_vector_v

            q_checkpoint = model.model.layers[i].self_attn.q_proj
            k_checkpoint = model.model.layers[i].self_attn.k_proj
            v_checkpoint = model.model.layers[i].self_attn.v_proj
            orig_norm_checkpoint = model.model.layers[i].self_attn.q_norm
            k_orig_norm_checkpoint = model.model.layers[i].self_attn.k_norm.weight.data

            k_layer_w = k_checkpoint.weight.data
            num_repeats = k_layer_w.size(0) // k_orig_norm_checkpoint.size(0) 
            k_w_checkpoint = k_layer_w * k_orig_norm_checkpoint.repeat(num_repeats).unsqueeze(1)

            kv_checkpoint = torch.cat([k_w_checkpoint, v_checkpoint.weight.data.float()])
            temp_layer = torch.nn.Linear(kv_checkpoint.shape[1], kv_checkpoint.shape[0], bias=False)
            temp_layer.weight.data = kv_checkpoint

            temp_identity_norm = type('', (), {})()
            temp_identity_norm.weight = type('', (), {})() 
            temp_identity_norm.weight.data = torch.ones(1, device=k_layer_w.device)

            q_d, q_w = svd_init_latent(
                module.self_attn.q_a, module.self_attn.q_b, orig_weights=q_checkpoint, rank=config.q_lora_rank,
                orig_norm=orig_norm_checkpoint, norm_layer=module.self_attn.q_norm_latent, 
                fisher_vector=fisher_vector_q, use_fisher=use_fisher, debug_information=True
            )
            
            kv_d, kv_w = svd_init_latent(
                module.self_attn.kv_a, module.self_attn.kv_b, orig_weights=temp_layer, rank=config.kv_lora_rank,
                orig_norm=temp_identity_norm, norm_layer=module.self_attn.kv_norm, 
                fisher_vector=fisher_vector_kv, use_fisher=use_fisher, debug_information=True
            )
            
            stats['q_diff'].append(q_d)
            stats['q_w_err'].append(q_w)
            stats['kv_diff'].append(kv_d)
            stats['kv_w_err'].append(kv_w)
            
    return stats

def plot_svd_ablation(fisher_stats, normal_stats):
    print("\nGenerating Ablation Plots...")
    layers = np.arange(len(fisher_stats['q_diff']))
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MaximusLLM Initialization: Fisher-SVD vs Standard SVD", fontsize=16, fontweight='bold')
    
    axs[0, 0].plot(layers, normal_stats['q_diff'], label='Standard SVD', marker='o', color='gray')
    axs[0, 0].plot(layers, fisher_stats['q_diff'], label='Fisher SVD', marker='s', color='blue')
    axs[0, 0].set_title("Query Proj: Standard L2 Error\n(Lower is better)")
    axs[0, 0].set_ylabel("Reconstruction Error")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(layers, normal_stats['q_w_err'], label='Standard SVD', marker='o', color='gray')
    axs[0, 1].plot(layers, fisher_stats['q_w_err'], label='Fisher SVD', marker='s', color='green')
    axs[0, 1].set_title("Query Proj: Semantic (Fisher) Error\n(Lower is better)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(layers, normal_stats['kv_diff'], label='Standard SVD', marker='o', color='gray')
    axs[1, 0].plot(layers, fisher_stats['kv_diff'], label='Fisher SVD', marker='s', color='blue')
    axs[1, 0].set_title("KV Proj: Standard L2 Error")
    axs[1, 0].set_xlabel("Layer Index")
    axs[1, 0].set_ylabel("Reconstruction Error")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(layers, normal_stats['kv_w_err'], label='Standard SVD', marker='o', color='gray')
    axs[1, 1].plot(layers, fisher_stats['kv_w_err'], label='Fisher SVD', marker='s', color='green')
    axs[1, 1].set_title("KV Proj: Semantic (Fisher) Error")
    axs[1, 1].set_xlabel("Layer Index")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svd_ablation_results.png', dpi=300)
    print("Plot saved to 'svd_ablation_results.png'")

print(">>> Computing Base Fisher Vectors...")
fisher_vector_global = compute_fisher_importance(model, num_batches=128)

config = Config.from_pretrained("yousefg/MaximusLLM")

print("\n>>> Running Fisher-SVD Initialization...")
model_fisher = Model(config, "cuda")
stats_fisher = test_fisher(model_fisher, fisher_vector_global, use_fisher=True)

print("\n>>> Running Standard SVD Initialization...")
model_normal = Model(config, "cuda")

stats_normal = test_fisher(model_normal, fisher_vector_global, use_fisher=False)

plot_svd_ablation(stats_fisher, stats_normal)

results = {
    "fisher_svd:": stats_fisher,
    "svd": stats_normal
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
