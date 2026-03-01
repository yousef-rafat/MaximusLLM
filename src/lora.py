import torch
import math
import torch.nn as nn
from model import Attention
import torch.nn.functional as F

# query, key -> ElongatingLoRALayer
# value, out -> NormalLora

class NormalLora(nn.Module):
    def __init__(self, original_layer, rank, alpha, num_heads, head_dim):
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_a = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, original_layer.out_features, bias=False)

        self.num_heads = num_heads
        self.head_dim = head_dim
        
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        base_out = self.original_layer(x)
        lora_out = self.lora_b(self.lora_a(x)) * self.scaling
        
        return base_out + lora_out

# long context lora
class ElongatingLoRALayer(NormalLora):
    def __init__(self, original_layer, rank, alpha, num_heads, head_dim):
        super().__init__(original_layer, rank, alpha, num_heads, head_dim)
        self.head_focus_scale = nn.Parameter(torch.ones(1, 1, num_heads, 1))

    def forward(self, x):
        output = super().forward(x)
        
        b, s, _ = output.shape
        output = output.view(b, s, self.num_heads, self.head_dim)
        
        output = output * self.head_focus_scale
        
        output = output.reshape(b, s, -1)
        
        return output

def get_dct_orthonormal_init(rows, cols, sink_size=4, freq_sink_size=8):

    i = torch.arange(rows).reshape(rows, 1)
    j = torch.arange(cols).reshape(1, cols)

    dct = torch.cos(torch.pi / cols * (j + 0.5) * i)

    # normalize
    dct[0, :] *= math.sqrt(1.0 / cols)
    dct[1:, :] *= math.sqrt(2.0 / cols)

    # rademacher dist. kind of (not to bias the model)
    random_signs = torch.randint(0, 2, (1, cols)).float() * 2.0 - 1.0
    spectral_ortho = dct * random_signs
    
    # boost sink tokens
    spectral_ortho[:, :sink_size] *= 2.0

    # boost low freq. rows
    spectral_ortho[:freq_sink_size, :] *= 1.5 
    return spectral_ortho

class RandNLAGQALayer(nn.Module):
    def __init__(self, original_layer: Attention, sketch_size=640, topk_size=2048):
        super().__init__()
        self.target_layer = original_layer
        self.max_context = 33 * 1024 
        self.sketch_size = sketch_size

        if hasattr(original_layer, 'config'):
            hidden_size = original_layer.config.hidden_size
        else:
            hidden_size = original_layer.q_proj.in_features

        self.num_heads = original_layer.num_heads 
        self.num_kv_heads = original_layer.num_key_value_heads
        self.head_dim = original_layer.head_dim
        self.topk_size = topk_size
        
        #  (640, 32768)
        self.kron_a = nn.Parameter(get_dct_orthonormal_init(20, 128))
        self.kron_b = nn.Parameter(get_dct_orthonormal_init(32, 256))

        # to allow for causality
        mask_b = torch.zeros_like(self.kron_b)
        ratio_b = self.kron_b.shape[1] / self.kron_b.shape[0]
        for i in range(self.kron_b.shape[0]):
            mask_b[i, :int((i + 1) * ratio_b)] = 1.0
        self.register_buffer("kron_b_mask", mask_b)
        
        mask_a = torch.zeros_like(self.kron_a)
        ratio_a = self.kron_a.shape[1] / self.kron_a.shape[0]
        for i in range(self.kron_a.shape[0]):
            mask_a[i, :int((i + 1) * ratio_a)] = 1.0
        self.register_buffer("kron_a_mask", mask_a)

        self.sketch_scale = nn.Parameter(torch.tensor([1.0]))

        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    # use the kronecker product identity to save on computing the full P matrix
    def apply_sketch(self, x):
        B, H, D, T = x.shape
        
        # 128 * 256
        target_len = self.kron_a.shape[-1] * self.kron_b.shape[-1]
        if T < target_len:
            x = F.pad(x, (0, target_len - T))
        
        x = x.view(B, H, D, 128, 256)

        # make kron a lower-trinagular (for autogressive-model compatibility)
        causal_kron_b = self.kron_b * self.kron_b_mask
        causal_kron_a = self.kron_a * self.kron_a_mask
        
        x = torch.matmul(x, causal_kron_b.t())
        x = x.transpose(-1, -2)
        x = torch.matmul(x, causal_kron_a.t())
        
        return x.flatten(-2)

    def get_importance_weights(self, x):
        _, seq_len, _ = x.shape
        importance_logits = self.importance_scorer(x)
        
        # put pressure
        if seq_len > self.sketch_size:
            compression_ratio = seq_len / self.sketch_size
            pressure_bias = math.log(compression_ratio)
            importance_logits = importance_logits - pressure_bias

        importance_weights = torch.sigmoid(importance_logits)
        return importance_weights, importance_logits
    
    def compute_efficient_oproj(self, output_full):
        chunks = torch.split(output_full, 1024, dim=1)
        results = []
        for chunk in chunks:
            results.append(self.target_layer.o_proj(chunk))
        return torch.cat(results, dim=1)
    
    def sketch_tensor(self, tensor, weights):
        t = (tensor * weights.unsqueeze(-1)).permute(0, 2, 3, 1)
        return self.apply_sketch(t).permute(0, 3, 1, 2)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        x = hidden_states
        bsz, seq_len, _ = x.shape
        cos, sin = self.target_layer.compute_freq_gl(position_embeddings)

        importance_weights, importance_logits = self.get_importance_weights(x)

        q = self.target_layer.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.target_layer.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.target_layer.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.target_layer.q_norm(q)
        k = self.target_layer.k_norm(k)
        
        q_rope = self.target_layer.rope_fn(q, cos, sin)
        actual_topk = min(seq_len, self.topk_size)
        _, topk_indices = torch.topk(importance_logits.squeeze(-1), actual_topk, dim=1)
        
        is_topk = torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool)
        is_topk.scatter_(1, topk_indices, True)
        is_rest = ~is_topk

        def batch_gather_heads(tensor, indices):
            H_dim, D_dim = tensor.shape[2], tensor.shape[3]
            return torch.gather(tensor, 1, indices.view(bsz, actual_topk, 1, 1).expand(-1, -1, H_dim, D_dim))

        k_detail = batch_gather_heads(k, topk_indices)
        v_detail = batch_gather_heads(v, topk_indices)
        
        def batch_gather_rope(tensor, indices):
            if tensor.size(0) != bsz:
                tensor = tensor.expand(bsz, -1, -1, -1)
            return torch.gather(tensor, 1, indices.view(bsz, actual_topk, 1, 1).expand(-1, -1, 1, tensor.shape[-1]))
            
        cos_detail = batch_gather_rope(cos, topk_indices)
        sin_detail = batch_gather_rope(sin, topk_indices)
        
        k_detail_rope = self.target_layer.rope_fn(k_detail, cos_detail, sin_detail)
        
        mask_float = is_rest.unsqueeze(-1).to(importance_weights.dtype)
        rest_weights = importance_weights * mask_float

        k_sketch = self.sketch_tensor(k, rest_weights) * self.sketch_scale
        v_sketch = self.sketch_tensor(v, rest_weights) * self.sketch_scale

        k_final = torch.cat([k_detail_rope, k_sketch], dim=1)
        v_final = torch.cat([v_detail, v_sketch], dim=1)

        k_final = torch.repeat_interleave(k_final, self.num_heads // self.num_kv_heads, dim=2).reshape(bsz, -1, self.num_heads, self.head_dim)
        v_final = torch.repeat_interleave(v_final, self.num_heads // self.num_kv_heads, dim=2).reshape(bsz, -1, self.num_heads, self.head_dim)

        q_time = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)
        
        k_time_detail = topk_indices.unsqueeze(1)
        mask_detail = k_time_detail <= q_time
        
        ratio = (self.kron_a.shape[1] * self.kron_b.shape[1]) / self.sketch_size
        sketch_times = (torch.arange(self.sketch_size, device=x.device) * ratio).view(1, 1, self.sketch_size)
        mask_sketch = (sketch_times <= q_time).expand(bsz, seq_len, self.sketch_size)
        
        attn_mask = torch.cat([mask_detail, mask_sketch], dim=-1).unsqueeze(1)

        output_chunks = []
        q_chunk_size = 1024 
        for i in range(0, seq_len, q_chunk_size):
            q_chunk = q_rope[:, :, i : i + q_chunk_size, :]
            mask_chunk = attn_mask[:, :, i : i + q_chunk_size, :]
            
            out_chunk = nn.functional.scaled_dot_product_attention(
                q_chunk, k_final, v_final, 
                attn_mask=mask_chunk, is_causal=False
            )
            output_chunks.append(out_chunk)

        attn_output = torch.cat(output_chunks, dim=2)

        output_full = attn_output.flatten(2)

        if self.training and seq_len > 2048:
            return self.compute_efficient_oproj(output_full)
        else: 
            return self.target_layer.o_proj(output_full)

class RandNLALatentAttention(RandNLAGQALayer):
    def __init__(self, original_layer: Attention, sketch_size=640):
        super().__init__(original_layer, sketch_size)
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        x = hidden_states
        bsz, seq_len, _ = x.shape
        cos, sin = self.target_layer.compute_freq_gl(position_embeddings)
        
        importance_weights, importance_logits = self.get_importance_weights(x)
        
        c_q = self.target_layer.q_norm_latent(self.target_layer.q_a(x))
        c_kv = self.target_layer.kv_norm(self.target_layer.kv_a(x))

        actual_topk = min(seq_len, self.topk_size)
        _, topk_indices = torch.topk(importance_logits.squeeze(-1), actual_topk, dim=1)
        
        is_topk = torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool)
        is_topk.scatter_(1, topk_indices, True)
        is_rest = ~is_topk

        q_all = self.target_layer.q_b(c_q).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        alpha = torch.sigmoid(self.target_layer.nope_logit)
        q_pe = self.target_layer.rope_fn(q_all, cos, sin)
        q_final = (alpha * q_all) + ((1 - alpha) * q_pe)
        
        q_final = q_final.transpose(1, 2)

        def batch_gather(tensor, indices):
            return torch.gather(tensor, 1, indices.unsqueeze(-1).expand(-1, -1, tensor.size(-1)))
        
        c_kv_detail = batch_gather(c_kv, topk_indices)
        
        mask_float = is_rest.unsqueeze(-1).to(importance_weights.dtype)
        rest_weights = importance_weights * mask_float

        c_kv_sketch = self.sketch_tensor(c_kv, rest_weights) * self.sketch_scale
        c_kv_final = torch.cat([c_kv_detail, c_kv_sketch], dim=1)
        
        kv_out = self.target_layer.kv_b(c_kv_final)
        k_out, v_out = torch.chunk(kv_out, 2, dim=-1)
        
        k_out = k_out.view(bsz, -1, self.num_kv_heads, self.head_dim)
        v_out = v_out.view(bsz, -1, self.num_kv_heads, self.head_dim)
        
        k_detail = k_out[:, :actual_topk]
        k_sketch = k_out[:, actual_topk:]
        
        def batch_gather_rope(tensor, indices):
            if tensor.size(0) != bsz:
                tensor = tensor.expand(bsz, -1, -1, -1)
            return torch.gather(tensor, 1, indices.view(bsz, actual_topk, 1, 1).expand(-1, -1, 1, tensor.shape[-1]))

        cos_det = batch_gather_rope(cos, topk_indices)
        sin_det = batch_gather_rope(sin, topk_indices)
        
        k_detail_pe = self.target_layer.rope_fn(k_detail, cos_det, sin_det)
        k_detail_final = (alpha * k_detail) + ((1 - alpha) * k_detail_pe)
        
        k_sketch_final = k_sketch * alpha
        
        k_final = torch.cat([k_detail_final, k_sketch_final], dim=1)
        v_final = v_out

        k_final = torch.repeat_interleave(k_final, self.num_heads // self.num_kv_heads, dim=2).reshape(bsz, -1, self.num_heads, self.head_dim)
        v_final = torch.repeat_interleave(v_final, self.num_heads // self.num_kv_heads, dim=2).reshape(bsz, -1, self.num_heads, self.head_dim)
            
        k_final = k_final.transpose(1, 2)
        v_final = v_final.transpose(1, 2)

        q_time = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)
        k_time_detail = topk_indices.unsqueeze(1)
        mask_detail = k_time_detail <= q_time
        
        ratio = seq_len / self.sketch_size
        sketch_times = (torch.arange(self.sketch_size, device=x.device) * ratio).view(1, 1, self.sketch_size)
        mask_sketch = (sketch_times <= q_time).expand(bsz, seq_len, self.sketch_size)
        
        attn_mask = torch.cat([mask_detail, mask_sketch], dim=-1).unsqueeze(1)

        output_chunks = []
        q_chunk_size = 1024 
        
        for i in range(0, seq_len, q_chunk_size):
            q_chunk = q_final[:, :, i : i + q_chunk_size, :]
            mask_chunk = attn_mask[:, :, i : i + q_chunk_size, :]
            
            out_chunk = nn.functional.scaled_dot_product_attention(
                q_chunk, k_final, v_final, 
                attn_mask=mask_chunk, is_causal=False
            )
            output_chunks.append(out_chunk)

        attn_output = torch.cat(output_chunks, dim=2)
        output_full = attn_output.transpose(1, 2).flatten(2)

        if self.training and seq_len > 2048:
            return self.compute_efficient_oproj(output_full)
        else: 
            return self.target_layer.o_proj(output_full)
    
def blockswap_attention_layers(model, sketch_size=640):
    for layer in model.layers:
        attn = layer.self_attn
        if attn.is_latent:
            layer.self_attn = RandNLALatentAttention(layer.self_attn, sketch_size)
        else:
            layer.self_attn = RandNLAGQALayer(layer.self_attn, sketch_size)

def blockswap(model, rank, alpha, num_heads, head_dim):
    args = [rank, alpha, num_heads, head_dim]
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn

        # full attn
        if hasattr(attn, "q_proj"):
            layer.self_attn.q_proj = ElongatingLoRALayer(layer.self_attn.q_proj, *args)
        if hasattr(attn, "k_proj"):
            layer.self_attn.k_proj = ElongatingLoRALayer(layer.self_attn.k_proj, *args)    
        if hasattr(attn, "v_proj"):
            layer.self_attn.v_proj = NormalLora(layer.self_attn.v_proj, *args)

        # latent attn
        if hasattr(attn, "q_a"):
            layer.self_attn.q_a = ElongatingLoRALayer(layer.self_attn.q_a, *args)
            layer.self_attn.q_b = ElongatingLoRALayer(layer.self_attn.q_b, *args)
        
        if hasattr(attn, "kv_a"):
            layer.self_attn.kv_a = NormalLora(layer.self_attn.kv_a, *args)
            layer.self_attn.kv_b = NormalLora(layer.self_attn.kv_b, *args)
    
        layer.self_attn.o_proj = NormalLora(layer.self_attn.o_proj, *args)
