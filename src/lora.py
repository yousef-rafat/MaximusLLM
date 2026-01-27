import torch
import torch.nn as nn
import math

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
