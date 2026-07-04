"""
ModelConfig — single source of truth for all model dimensions.
All layers read dimensions from this config; no magic numbers anywhere.
When porting to C++, this becomes a `struct ModelConfig { int ... ; };`.
"""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """All dimensions in one place. C++ struct equivalent."""
    vocab_size: int = 5000
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048       # 4 * d_model by default
    max_seq_len: int = 128
    dropout: float = 0.1
    use_rope: bool = True  # RoPE vs Sinusoidal
    use_bias: bool = True

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, \
            f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
        self.head_dim: int = self.d_model // self.num_heads
