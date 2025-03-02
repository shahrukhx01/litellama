from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from litellama.models.base_model import BaseModel
from litellama.models.gemma.gemma_config import GemmaConfig


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    """Precomputes positional frequencies for rotary embeddings.

    Args:
        head_dim (int): Dimension of the attention head.
        seq_len (int): Sequence length.
        device (str): Device to store the tensor.
        theta (float, optional): Scaling factor for position encoding. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed positional frequency tensor.
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(0, seq_len, device=device)
    freqs = torch.outer(m, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_embeddings(
    x: torch.Tensor, freqs_complex: torch.Tensor, device: str
) -> torch.Tensor:
    """Applies rotary positional embeddings to input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        freqs_complex (torch.Tensor): Precomputed frequency tensor.
        device (str): Device to store the tensor.

    Returns:
        torch.Tensor: Tensor with applied rotary embeddings.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex[None, :, None, :]
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initializes RMSNorm.

        Args:
            hidden_size (int): Size of the hidden layer.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes RMS normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after normalization.
        """
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key-value tensor to match the number of attention heads.

    Args:
        x (torch.Tensor): Input tensor.
        n_rep (int): Number of repetitions.

    Returns:
        torch.Tensor: Tensor with repeated key-value heads.
    """
    if n_rep == 1:
        return x
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention mechanism with rotary embeddings."""

    def __init__(self, config: GemmaConfig):
        """Initializes causal self-attention module.

        Args:
            config (GemmaConfig): Configuration object containing model parameters.
        """
        super(CausalSelfAttention, self).__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_attention_heads, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads, config.hidden_size, bias=False
        )

        self.cache_k = torch.zeros(
            (
                config.batch_size,
                config.max_position_embeddings,
                self.num_key_value_heads,
                self.head_dim,
            ),
            device=config.device,
        )
        self.cache_v = torch.zeros(
            (
                config.batch_size,
                config.max_position_embeddings,
                self.num_key_value_heads,
                self.head_dim,
            ),
            device=config.device,
        )

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Start position for rotary embeddings.
            freqs_complex (torch.Tensor): Precomputed positional frequencies.

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
        """
        batch_size, seq_len, hidden_size = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk  # noqa
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv  # noqa
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        queries, keys, values = (
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
        )
        attn = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim**0.5
        attn = torch.nn.functional.softmax(attn, dim=-1).type_as(xq)
        out = (
            torch.matmul(attn, values)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        return self.o_proj(out)


class GemmaMLP(nn.Module):
    """Gemma Multi-Layer Perceptron (MLP) block.

    Args:
        config (GemmaConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: GemmaConfig):
        super(GemmaMLP, self).__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed output tensor.
        """
        x_swish = nn.functional.gelu(self.gate_proj(x))
        x_V = self.up_proj(x)
        x = x_swish * x_V
        return self.down_proj(x)


class Block(nn.Module):
    """Transformer block consisting of self-attention and MLP layers.

    Args:
        config (GemmaConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: GemmaConfig):
        super(Block, self).__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Start position for rotary embeddings.
            freqs_complex (torch.Tensor): Precomputed rotary embedding frequencies.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and MLP layers.
        """
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_complex)
        return h + self.mlp(self.post_attention_layernorm(h))


class GemmaModel(nn.Module):
    """Gemma base model implementing the core transformer architecture.

    Args:
        config (GemmaConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.freqs_complex = precompute_theta_pos_frequencies(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
            device=config.device,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of token ids.
            start_pos (int): Starting position for position embeddings.
            freqs_complex (Optional[torch.Tensor]): Optional precomputed frequency tensors for rotary embeddings.

        Returns:
            torch.Tensor: Output hidden states.
        """
        _, seq_len = x.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        x = self.embed_tokens(x)
        freqs_complex = (
            freqs_complex
            if freqs_complex is not None
            else self.freqs_complex[start_pos : start_pos + seq_len]  # noqa
        )

        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)

        return self.norm(x)


class GemmaCausalLM(BaseModel):
    """Gemma Language Model implementing an autoregressive decoder-only architecture.

    Args:
        config (GemmaConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config.name_or_path, config.device)
        assert config.vocab_size != -1, "Vocab size must be set"
        self._config = config

        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        logger.info(
            f"Model initialized with the configuration of {config.name_or_path}"
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the causal language model.

        Args:
            x (torch.Tensor): Input tensor of token ids.
            start_pos (int): Starting position for position embeddings.
            freqs_complex (Optional[torch.Tensor]): Optional precomputed frequency tensors for rotary embeddings.

        Returns:
            torch.Tensor: Logits over vocabulary.
        """
        hidden_states = self.model(x, start_pos, freqs_complex)
        return self.lm_head(hidden_states)

    def load_pretrained_hf(self):
        """Load pretrained weights from a Hugging Face model checkpoint."""
        import time

        from transformers import GemmaForCausalLM

        load_start_ts = time.perf_counter()
        pretrained: GemmaForCausalLM = GemmaForCausalLM.from_pretrained(
            self._config.name_or_path
        )
        self.load_state_dict(pretrained.state_dict(), strict=True)
        load_duration = time.perf_counter() - load_start_ts
        logger.info(
            f"Successfully loaded weights from pretrained model: {self._config.name_or_path} in {load_duration:.2f}"
            " seconds."
        )
