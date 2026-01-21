import copy
import math
import torch
import torch.nn as nn
from transformers import DynamicCache, Gemma3TextConfig, AutoModelForCausalLM
from fisher_svd import svd_init_latent, compute_fisher_importance, model_id, token

def create_causal_padding_mask(attention_mask, seq_len, dtype, device):

    if dtype == torch.float16:
        min_val = -65500.0
    elif dtype == torch.bfloat16:
        min_val = -3.4e38
    else:
        min_val = -1e9

    if attention_mask.dim() > 3: # packing
        final_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)        
        final_mask = final_mask.masked_fill(attention_mask, min_val)
        return final_mask

    batch_size = attention_mask.shape[0]
    causal_mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool), 
        diagonal=1
    )
    
    padding_mask = (attention_mask[:, None, None, :] == 0)
    combined_mask = causal_mask | padding_mask


    final_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
    final_mask = final_mask.masked_fill(combined_mask, min_val)
    
    return final_mask  

class EmbeddingWithScale(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0, device=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, device=device)
        self.register_buffer("embed_scale", torch.tensor(embed_scale, device=device), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

class MLP(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, config: "Config", device):
        super().__init__()
        self.device = device
        self.head_dim = config.head_dim
        self.base = config.base
        self.config = config

        self.initial_context_length = config.initial_context_length
        self.fast_beta = config.fast_beta
        self.slow_beta = config.slow_beta
        self.ntk_alpha = config.ntk_alpha
        self.use_yarn = config.use_yarn

    # dynamic ntk
    def _compute_inv_freq_dynamic(self, seq_len: int):
        freq = self.base ** (torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim)
        inv_freq = 1.0 / freq
        scale = (seq_len / self.initial_context_length) ** self.ntk_alpha
        inv_freq = inv_freq / scale
        concentration = 1.0
        return concentration, inv_freq

    # yarn
    def _compute_inv_freq_yarn(self, seq_len, scale_factor=None):

        if scale_factor is None:
            scale_factor = max(1.0, seq_len / self.initial_context_length)

        dim = self.head_dim
        freqs = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        
        wavelen = 2 * math.pi / freqs
        
        ramp = (wavelen - self.slow_beta) / (self.fast_beta - self.slow_beta)
        ramp = torch.clamp(ramp, 0.0, 1.0)

        inv_freq_interpolated = freqs / scale_factor 
        inv_freq = (1 - ramp) * freqs + ramp * inv_freq_interpolated
        
        if scale_factor > 1.0:
            mscale = 0.1 * math.log(scale_factor) + 1.0
            concentration = mscale
        else:
            concentration = 1.0

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens):
        if self.config.use_yarn:
            concentration, inv_freq = self._compute_inv_freq_yarn(num_tokens)
        else:
            concentration, inv_freq = self._compute_inv_freq_dynamic(num_tokens)
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

class Attention(nn.Module):

    def __init__(self, config, layer_idx: int, device):
        super().__init__()
        self.is_latent = layer_idx not in [5, 11, config.num_hidden_layers - 1]
        self.config = config
        self.layer_idx = layer_idx

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.attention_dropout = self.config.attention_dropout
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads

        if not self.is_latent: # GQA Attention

            self.scaling = config.query_pre_attn_scalar ** -0.5

            self.q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, device=device
            )
            self.k_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device=device
            )
            self.v_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device=device
            )

            self.k_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps, device=device)
            self.q_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps, device=device)

        else:

            self.scaling = self.head_dim ** -0.5

            self.q_a = nn.Linear(config.hidden_size, config.q_lora_rank, device=device)
            self.q_b = nn.Linear(config.q_lora_rank, self.head_dim * config.num_attention_heads, device=device)

            self.kv_a = nn.Linear(config.hidden_size, config.kv_lora_rank, device=device)
            self.kv_b = nn.Linear(config.kv_lora_rank, (config.num_key_value_heads * self.head_dim) * 2, device=device)

            self.kv_norm = RMSNorm(dim = config.kv_lora_rank, eps = config.rms_norm_eps, device=device)
            self.q_norm_latent = RMSNorm(dim = config.q_lora_rank, eps = config.rms_norm_eps, device=device)


        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, device=device
        )

        if self.is_latent:
            # start with strong RoPE bias
            self.nope_logit = nn.Parameter(torch.tensor(-3.0))

        rope_logit = 0.5 # start with equal local and global rope
        logit = torch.log(torch.tensor(rope_logit) / (1.0 - torch.tensor(rope_logit)))
        self.rope_global_local_logit = nn.Parameter(logit)

    def compute_freq_gl(self, pos_embed):
        cos_g, sin_g, cos_l, sin_l = pos_embed
        alpha = torch.sigmoid(self.rope_global_local_logit)
        alpha = alpha.view(1, 1, 1, 1)
        rope_dim = self.head_dim // 2

        cos = cos_g.view(1, -1, 1, rope_dim) * alpha + (1.0 - alpha) * cos_l.view(1, -1, 1, rope_dim)
        sin = sin_g.view(1, -1, 1, rope_dim) * alpha + (1.0 - alpha) * sin_l.view(1, -1, 1, rope_dim)

        return cos, sin

    def apply_latent_attention(self, hidden_states, freqs_cis):

        cos, sin = freqs_cis

        # query
        q = self.q_b(self.q_norm_latent(self.q_a(hidden_states)))
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim)
        q_pe = _apply_rotary_emb(q, cos, sin)

        # key & value
        kv = self.kv_b(self.kv_norm(self.kv_a(hidden_states)))
        k, v_final = torch.chunk(kv, 2, dim = -1)
        k = k.reshape(q.size(0), -1, self.num_key_value_heads, self.head_dim).contiguous()
        k_pe = _apply_rotary_emb(k, cos, sin)

        alpha = torch.sigmoid(self.nope_logit)

        # interpolate between rope and nope
        q_final = alpha * q + (1 - alpha) * q_pe
        k_final = alpha * k + (1 - alpha) * k_pe

        return q_final, k_final, v_final

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        past_key_values = None,
        cache_position = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = self.compute_freq_gl(position_embeddings)

        if not self.is_latent:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        else:
            query_states, key_states, value_states = self.apply_latent_attention(hidden_states, (cos, sin))

        if not self.is_latent:
            query_states = _apply_rotary_emb(query_states, cos, sin)
            key_states = _apply_rotary_emb(key_states, cos, sin)

        if value_states.ndim == 3:
            value_states = value_states.unsqueeze(2)
        query_states, key_states, value_states = [t.transpose(1, 2) for t in (query_states, key_states, value_states)]

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if attention_mask is not None:
            attention_mask = create_causal_padding_mask(
                attention_mask, query_states.size(2), device=query_states.device, dtype=query_states.dtype
            )
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask, is_causal=False)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, device):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config=config, layer_idx=layer_idx, device= device)
        self.mlp = MLP(config, device=device)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, device=device)
        self.pre_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, device=device)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        position_ids= None,
        past_key_values = None,
        output_attentions= False,
        use_cache = False,
        cache_position= None,
        **kwargs,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask = attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs

class Model(nn.Module):

    def __init__(self, config, device="cpu"):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads


        self.embed_tokens = EmbeddingWithScale(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5, device=device
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx, device) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device = device)
        self.rotary_emb = RotaryEmbedding(config=config, device = device)

        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        self.rotary_emb_local = RotaryEmbedding(config=config, device = device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta")
        self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False

    def init_latent_attention(self, num_batches=100):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, torch_dtype=torch.float32
        )
        fisher_vector = compute_fisher_importance(model, num_batches=num_batches)
        for i, module in enumerate(self.layers.modules()):
            if hasattr(module.self_attn, "q_a"):
                q_checkpoint = model.self_attn.q_proj
                k_checkpoint = model.self_attn.k_proj
                v_checkpoint = model.self_attn.v_proj
                orig_norm_checkpoint = model.self_attn.q_norm
                k_orig_norm_checkpoint = model.self_attn.k_norm
                kv_checkpoint = torch.cat([k_checkpoint, v_checkpoint])
                temp_layer = nn.Linear(kv_checkpoint.shape[1], kv_checkpoint.shape[0], bias=False)
                temp_layer.weight.data = kv_checkpoint
                svd_init_latent(
                    module.self_attn.q_a, module.self_attn.q_b, orig_weights=q_checkpoint, rank=self.config.q_lora_rank,
                    orig_norm=orig_norm_checkpoint, norm_layer=module.self_attn.q_norm_latent, fisher_vector = fisher_vector
                )
                svd_init_latent(
                    module.self_attn.kv_a, module.self_attn.kv_b, orig_weights=temp_layer, rank=self.config.kv_lora_rank,
                    orig_norm=k_orig_norm_checkpoint, norm_layer=module.self_attn.k_norm, fisher_vector = fisher_vector
                )

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        cache_position = None,
        return_hidden = False,
        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if use_cache:
            seq_len = cache_position.max().item() + 1
        else:
            seq_len = input_ids.size(1)

        # mix the global and local cos/sin 
        cos_g, sin_g = self.rotary_emb._compute_cos_sin(seq_len)
        cos_l, sin_l = self.rotary_emb_local._compute_cos_sin(seq_len)

        # inference
        if use_cache:
            cos_g = cos_g[cache_position]
            sin_g = sin_g[cache_position]
            cos_l = cos_l[cache_position]
            sin_l = sin_l[cache_position]

        positional_embeddings = (cos_g, sin_g, cos_l, sin_l)

        hidden_states = inputs_embeds
        num_layers = self.config.num_hidden_layers
        model_args = {
            "attention_mask": attention_mask,
            "position_embeddings": positional_embeddings,
            "past_key_values":past_key_values,
            "use_cache":use_cache,
            "cache_position":cache_position,
            **kwargs
        }
        for i, decoder_layer in enumerate(self.layers[: num_layers]):
            
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(decoder_layer, hidden_states=hidden_states, **model_args, use_reentrant = False)
            else:
                hidden_states = decoder_layer(hidden_states=hidden_states, **model_args)

        hidden_states = self.norm(hidden_states)
        if return_hidden:
            return hidden_states
        logits = self.lm_head(hidden_states)

        if use_cache:
            return logits, past_key_values
        return logits

# TODO: move to a json file
class Config(Gemma3TextConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_lora_rank = 128
        self.kv_lora_rank = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = 10_000
        self.initial_context_length = 4096
        self.fast_beta = 32
        self.slow_beta = 1
        self.use_yarn = False # for long context training only
        self.ntk_alpha = 1.0
