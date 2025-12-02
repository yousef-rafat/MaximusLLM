import copy
import torch
from typing import *
import torch.nn as nn
from transformers import DynamicCache, Cache, Gemma3TextConfig 

class EmbeddingWithScale(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

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
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.base
        self.initial_context_length = config.initial_context_length
        self.ntk_alpha = config.ntk_alpha
        self.device = config.device

    # dynamic ntk
    def _compute_inv_freq_dynamic(self, seq_len: int):
        freq = self.base ** (torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim)
        inv_freq = 1.0 / freq
        scale = (seq_len / self.initial_context_length) ** self.ntk_alpha
        inv_freq = inv_freq / scale
        concentration = 1.0
        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens):
        concentration, inv_freq = self._compute_inv_freq_dynamic(num_tokens)
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

class Attention(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.is_latent = not (layer_idx in [0, config.num_hidden_layers // 2, config.num_hidden_layers])
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
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
            )
            self.k_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )

            self.k_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps)
            self.q_norm = RMSNorm(dim = config.head_dim, eps = config.rms_norm_eps)

        else:

            self.scaling = self.head_dim ** -0.5

            self.q_a = nn.Linear(config.hidden_size, config.q_lora_rank)
            self.q_b = nn.Linear(config.q_lora_rank, self.head_dim * config.num_attention_heads)

            self.kv_a = nn.Linear(config.hidden_size, config.kv_lora_rank)
            self.kv_b = nn.Linear(config.kv_lora_rank, (config.num_key_value_heads * self.head_dim) * 2)

            self.kv_norm = RMSNorm(dim = config.kv_lora_rank, eps = config.rms_norm_eps)
            self.q_norm = RMSNorm(dim = config.q_lora_rank, eps = config.rms_norm_eps)


        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

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
        q = self.q_b(self.q_norm(self.q_a(hidden_states)))
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

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if value_states.ndim == 3:
            value_states = value_states.unsqueeze(2)

        attention_mask = attention_mask[:, :, None, None]
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_embeddings: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
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

        outputs = (hidden_states,)

        return outputs

class Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads


        self.embed_tokens = EmbeddingWithScale(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        self.rotary_emb_local = RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def post_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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

        seq_len = input_ids.size(1)

        # mix the global and local cos/sin 
        cos_g, sin_g = self.rotary_emb._compute_cos_sin(seq_len)
        cos_l, sin_l = self.rotary_emb_local._compute_cos_sin(seq_len)

        positional_embeddings = (cos_g, sin_g, cos_l, sin_l)

        hidden_states = inputs_embeds
        num_layers = self.config.num_hidden_layers
        for i, decoder_layer in enumerate(self.layers[: num_layers]):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_embeddings = positional_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

# TODO: move to a json file
class Config(Gemma3TextConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_lora_rank = 32 # may increase
        self.kv_lora_rank = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = 10_000
        self.initial_context_length = 4096
        self.ntk_alpha = 1.0

