"""
positional.py — Positional encodings.

Two modes (selectable via ModelConfig.use_rope):
  1. Sinusoidal (original transformer): fixed sin/cos per position
  2. RoPE (modern): applied inside attention, better for longer contexts

Both are stateless after init; pre-compute caches for speed.
C++: pre-allocate a float array for sin/cos at init.
"""

import numpy as np
from optimized_llm.config import ModelConfig


class SinusoidalEncoding:
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Pre-computed for max_seq_len.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.pe: np.ndarray = self._build(cfg.max_seq_len, cfg.d_model)

    @staticmethod
    def _build(max_seq_len: int, d_model: int) -> np.ndarray:
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
        pos = np.arange(max_seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32)
                          * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to embeddings.
        x: (batch, seq, d_model)
        Returns: (batch, seq, d_model)
        C++: for i in 0..seq: for j in 0..d_model: x[i][j] += pe[i][j]
        """
        return x + self.pe[:x.shape[1]]


class RoPE:
    """
    Rotary Position Embedding (Su et al. 2021).
    Applies rotation to Q and K vectors before attention.
    Pre-computes cos/sin caches for all positions.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.cos_cached, self.sin_cached = self._build_cache(cfg.max_seq_len, self.head_dim)

    @staticmethod
    def _build_cache(max_seq_len: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(max_seq_len, dtype=np.float32).reshape(-1, 1)
        angles = pos * inv_freq  # (max_seq_len, head_dim // 2)
        cos_cached = np.cos(angles)
        sin_cached = np.sin(angles)
        return cos_cached, sin_cached

    def forward(self, q: np.ndarray, k: np.ndarray,
                positions: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply RoPE to Q and K heads.
        q, k: (batch, num_heads, seq, head_dim)
        positions: optional int array of position indices
        Returns: rotated q, rotated k
        """
        seq_len = q.shape[2]
        hd = self.head_dim

        if positions is None:
            positions = np.arange(seq_len)

        cos = self.cos_cached[positions].reshape(1, 1, seq_len, hd // 2)
        sin = self.sin_cached[positions].reshape(1, 1, seq_len, hd // 2)

        q1, q2 = q[..., :hd // 2], q[..., hd // 2:]
        k1, k2 = k[..., :hd // 2], k[..., hd // 2:]

        q_rot = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rot = np.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        return q_rot, k_rot


def build_positional_encoding(cfg: ModelConfig):
    """Factory: returns SinusoidalEncoding or RoPE based on config."""
    if cfg.use_rope:
        return RoPE(cfg)
    return SinusoidalEncoding(cfg)
