"""
transformer_block.py — Transformer block with pre-norm architecture.

Architecture (pre-norm):
  x → LayerNorm → MultiHeadAttention → residual add → LayerNorm → FFN → residual add

Pre-norm is more stable and commonly used in modern LLMs (GPT, LLaMA style).
"""

import numpy as np
from optimized_llm.config import ModelConfig
from optimized_llm.tensor_ops import fused_layer_norm, fused_rms_norm
from optimized_llm.attention import MultiHeadAttention, KVCache
from optimized_llm.linear import FeedForward


class TransformerBlock:
    """
    Single transformer block with pre-norm architecture.
    C++: struct { LayerNorm ln1, ln2; MultiHeadAttention attn; FeedForward ffn; }
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.attention = MultiHeadAttention(cfg)
        self.ffn = FeedForward(cfg)

        # LayerNorm parameters
        self.gamma1 = np.ones(cfg.d_model, dtype=np.float32)
        self.beta1 = np.zeros(cfg.d_model, dtype=np.float32)
        self.gamma2 = np.ones(cfg.d_model, dtype=np.float32)
        self.beta2 = np.zeros(cfg.d_model, dtype=np.float32)

        self.last_x: np.ndarray = None  # input to block
        self.last_norm1_out: np.ndarray = None
        self.last_attn_out: np.ndarray = None
        self.last_residual1: np.ndarray = None
        self.last_norm2_out: np.ndarray = None
        self.last_ffn_out: np.ndarray = None
        self.last_residual2: np.ndarray = None
        self.last_grad_gamma1: np.ndarray = None
        self.last_grad_beta1: np.ndarray = None
        self.last_grad_gamma2: np.ndarray = None
        self.last_grad_beta2: np.ndarray = None

    def forward(self, x: np.ndarray, mask: np.ndarray = None,
                training: bool = False,
                use_cache: bool = False, kv_cache: KVCache = None) -> np.ndarray:
        """
        Pre-norm transformer block.
        x: (batch, seq, d_model)
        training: if True, enables dropout
        Returns: (batch, seq, d_model)
        """
        # --- Sub-layer 1: Attention with pre-norm ---
        self.last_x = x
        norm1_out = fused_layer_norm(x, self.gamma1, self.beta1)
        self.last_norm1_out = norm1_out
        self.last_residual1 = x

        attn_out = self.attention.forward(norm1_out, mask,
                                          use_cache=use_cache,
                                          kv_cache=kv_cache,
                                          layer_idx=self.layer_idx)
        self.last_attn_out = attn_out

        x = x + attn_out  # residual

        # --- Sub-layer 2: FFN with pre-norm ---
        norm2_out = fused_layer_norm(x, self.gamma2, self.beta2)
        self.last_norm2_out = norm2_out
        self.last_residual2 = x

        ffn_out = self.ffn.forward(norm2_out)
        self.last_ffn_out = ffn_out

        x = x + ffn_out  # residual
        return x

    def get_parameters(self) -> dict:
        """Return all trainable parameters for this block."""
        return {
            f'block_{self.layer_idx}_gamma1': self.gamma1,
            f'block_{self.layer_idx}_beta1': self.beta1,
            f'block_{self.layer_idx}_gamma2': self.gamma2,
            f'block_{self.layer_idx}_beta2': self.beta2,
        }
