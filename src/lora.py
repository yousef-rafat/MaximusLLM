import torch
import torch.nn as nn
import math
from model import Attention

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

class RandNLAGQALayer(nn.Module):
    def __init__(self, original_layer, sketch_size=640):
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
        
        #  (640, 32768)
        self.kron_a = nn.Parameter(torch.randn(20, 128))
        self.kron_b = nn.Parameter(torch.randn(32, 256))

        nn.init.orthogonal_(self.kron_a)
        nn.init.orthogonal_(self.kron_b)

        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_importance_weights(self, x):
        _, seq_len, _ = x.shape
        importance_logits = self.importance_scorer(x)
        
        # put pressure
        if seq_len > self.sketch_size:
            compression_ratio = seq_len / self.sketch_size
            pressure_bias = math.log(compression_ratio)
            importance_logits = importance_logits - pressure_bias

        importance_weights = torch.sigmoid(importance_logits)
        return importance_weights
    
    def get_p(self, seq_len):
        P = torch.kron(self.kron_a, self.kron_b)
        return P[:, :seq_len]

    def forward(self, x, attention_mask=None, positional_emb=None, **kwargs):
        bsz, seq_len, _ = x.shape
        cos, sin = positional_emb

        importance_weights = self.get_importance_weights(x)

        q = self.target_layer.q_proj(x)
        k = self.target_layer.k_proj(x)
        v = self.target_layer.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.target_layer.q_norm(q)
        k = self.target_layer.k_norm(k)

        P_curr = self.get_p(seq_len)

        k_weighted = k * importance_weights.unsqueeze(-1)
        v_weighted = v * importance_weights.unsqueeze(-1)

        k_t = k_weighted.permute(0, 2, 3, 1)
        v_t = v_weighted.permute(0, 2, 3, 1)
        
        k_sketched = torch.matmul(k_t, P_curr.t())
        v_sketched = torch.matmul(v_t, P_curr.t())
        
        k_final = k_sketched.permute(0, 3, 1, 2)
        v_final = v_sketched.permute(0, 3, 1, 2)
        
        q_pe = self.target_layer.rope_fn(q, cos, sin)

        cos_sketch = cos[:, :self.sketch_size, :, :]
        sin_sketch = sin[:, :self.sketch_size, :, :]
        k_pe = self.target_layer.rope_fn(k_final, cos_sketch, sin_sketch)

        k_pe = self.repeat_kv(k_pe, self.num_heads // self.num_kv_heads)
        v_final = self.repeat_kv(v_final, self.num_heads // self.num_kv_heads)

        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)
        v_final = v_final.transpose(1, 2)
        
        attn_output = nn.functional.scaled_dot_product_attention(
            q_pe, k_pe, v_final, 
            attn_mask=None, 
            is_causal=False
        )

        output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return self.target_layer.o_proj(output)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class RandNLALatentAttention(RandNLAGQALayer):
    def __init__(self, original_layer: Attention, sketch_size=640):
        super().__init__(original_layer, sketch_size)
    def forward(self, x, attention_mask=None, positional_emb=None, **kwargs):
        
        bsz, seq_len, _ = x.shape
        cos, sin = positional_emb
        importance_weights = self.get_importance_weights(x)
        
        c_q = self.target_layer.q_a(x)
        c_q = self.target_layer.q_norm_latent(c_q)
        
        q = self.target_layer.q_b(c_q)
        q = q.view(bsz, seq_len, self.target_layer.num_heads, self.target_layer.head_dim)
        
        q_pe = self.target_layer.rope_fn(q, cos, sin)

        c_kv = self.target_layer.kv_a(x)
        c_kv = self.target_layer.kv_norm(c_kv)

        c_kv = c_kv * importance_weights

        P_curr = self.get_p(seq_len)
        
        c_kv_t = c_kv.transpose(1, 2)
        c_kv_sketched = torch.matmul(c_kv_t, P_curr.t())
        c_kv_sketched = c_kv_sketched.transpose(1, 2)

        kv = self.target_layer.kv_b(c_kv_sketched)
        
        k, v_final = torch.chunk(kv, 2, dim=-1)
        k = k.reshape(bsz, self.sketch_size, self.target_layer.num_key_value_heads, self.target_layer.head_dim)
        v_final = v_final.reshape(bsz, self.sketch_size, self.target_layer.num_key_value_heads, self.target_layer.head_dim)

        alpha = torch.sigmoid(self.target_layer.nope_logit)
        
        q_final = alpha * q + (1 - alpha) * q_pe

        cos_sketch = cos[:, :self.sketch_size, :, :]
        sin_sketch = sin[:, :self.sketch_size, :, :]
        
        k_pe = self.target_layer.rope_fn(k, cos_sketch, sin_sketch)
        k_final = alpha * k + (1 - alpha) * k_pe

        q_final = q_final.transpose(1, 2)
        k_final = k_final.transpose(1, 2)
        v_final = v_final.transpose(1, 2)
        
        attn_output = nn.functional.scaled_dot_product_attention(q_final, k_final, v_final, attn_mask = None, is_causal=False)
        output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        
        return self.target_layer.o_proj(output)
    
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
