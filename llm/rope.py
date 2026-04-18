"""
RoPE (Rotary Positional Embedding) Module
More modern positional encoding that provides better performance
for attention-based models. RoPE encodes position information
through rotation matrices in the attention computation.
"""

import numpy as np


def get_rope_cache(seq_len, head_dim):
    """
    Precompute sin and cos values for RoPE.
    
    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension of each attention head
    
    Returns:
        cos_cached: Cosine values
        sin_cached: Sine values
    """
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    
    positions = np.arange(seq_len).reshape(-1, 1)
    angles = positions * inv_freq
    
    cos_cached = np.cos(angles)
    sin_cached = np.sin(angles)
    
    return cos_cached, sin_cached


def rotate_half(x):
    """
    Rotate half of the hidden units.
    
    Args:
        x: Input tensor (..., head_dim)
    
    Returns:
        rotated: Rotated tensor
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin, positions=None):
    """
    Apply RoPE to input tensor.
    
    Args:
        x: Input tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine values (seq_len, head_dim // 2)
        sin: Sine values (seq_len, head_dim // 2)
        positions: Optional positions to use (if None, use all)
    
    Returns:
        output: RoPE-encoded tensor
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    
    if positions is None:
        positions = np.arange(seq_len)
    
    cos = cos[positions]  # (seq_len, head_dim // 2)
    sin = sin[positions]  # (seq_len, head_dim // 2)
    
    cos = cos.reshape(1, 1, seq_len, head_dim // 2)
    sin = sin.reshape(1, 1, seq_len, head_dim // 2)
    
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    return np.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)


class RoPE:
    """
    Rotary Positional Embedding layer.
    """

    def __init__(self, max_seq_length, head_dim, max_positions=None):
        self.max_seq_length = max_seq_length
        self.head_dim = head_dim
        self.max_positions = max_positions or max_seq_length
        
        self.cos_cached, self.sin_cached = get_rope_cache(self.max_positions, head_dim)

    def forward(self, x, positions=None):
        """
        Apply RoPE to input.
        
        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim)
            positions: Optional positions
        
        Returns:
            output: RoPE-encoded tensor
        """
        return apply_rope(x, self.cos_cached, self.sin_cached, positions)

    def get_cache(self):
        """Get the cached sin/cos values."""
        return self.cos_cached, self.sin_cached

    def update_cache(self, new_seq_len):
        """Update cache for longer sequences."""
        if new_seq_len > self.max_positions:
            self.max_positions = new_seq_len
            self.cos_cached, self.sin_cached = get_rope_cache(new_seq_len, self.head_dim)


class RoPEEncoder:
    """
    Complete RoPE encoding layer that combines embedding and RoPE.
    """

    def __init__(self, d_model, max_seq_length, num_heads):
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.rope = RoPE(max_seq_length, self.head_dim)

    def forward(self, x, positions=None):
        """
        Apply RoPE to input embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            positions: Optional position indices
        
        Returns:
            output: RoPE-encoded tensor
        """
        batch_size, seq_len, d_model = x.shape
        
        x_reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x_transposed = x_reshaped.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        
        x_rope = self.rope.forward(x_transposed, positions)
        
        x_rope = x_rope.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        return x_rope


class RoPEMultiHeadAttention:
    """
    Multi-head attention with RoPE positional embeddings.
    """

    def __init__(self, d_model, num_heads, use_bias=True, max_seq_length=128):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = np.sqrt(self.head_dim)
        self.max_seq_length = max_seq_length

        from qkv import init_qkv_weights
        self.qkv_weights = init_qkv_weights(d_model, num_heads, use_bias)
        
        self.rope = RoPE(max_seq_length, self.head_dim)

    def forward(self, x, mask=None, positions=None):
        """
        Forward pass with RoPE.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional attention mask
            positions: Optional position indices
        
        Returns:
            output: Attention output
            attention_weights: Attention probabilities
        """
        from attention import compute_qkv, split_into_heads
        from softmax import scaled_softmax
        
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        
        batch_size, seq_len, d_model = x.shape
        
        q, k, v = compute_qkv(x, self.qkv_weights)
        
        q_heads, k_heads, v_heads = split_into_heads(q, k, v, self.num_heads)
        
        q_heads = self.rope.forward(q_heads, positions)
        k_heads = self.rope.forward(k_heads, positions)
        
        attention_scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            attention_scores = np.where(mask == False, -1e9, attention_scores)
        
        attention_weights = scaled_softmax(attention_scores, 1.0, axis=-1)
        
        output = np.matmul(attention_weights, v_heads)
        
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        return output, attention_weights


def create_rope_transformer_block(d_model, num_heads, d_ff=None, max_seq_length=128, dropout=0.0):
    """
    Create a transformer block with RoPE.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    
    Returns:
        block: Transformer block with RoPE
    """
    from transformer_block import TransformerBlock
    
    class RoPETransformerBlock(TransformerBlock):
        def __init__(self, d_model, num_heads, d_ff=None, max_seq_length=128, dropout=0.0):
            super().__init__(d_model, num_heads, d_ff, dropout)
            
            self.rope_encoder = RoPEEncoder(d_model, max_seq_length, num_heads)
            
            self.attention = RoPEMultiHeadAttention(d_model, num_heads, use_bias=True, max_seq_length=max_seq_length)
        
        def forward(self, x, mask=None, training=True):
            self.cache_x = x.copy()
            
            x_with_rope = self.rope_encoder.forward(x)
            
            attn_output, _ = self.attention.forward(x_with_rope, mask)

            if training and self.dropout > 0:
                mask_arr = np.random.rand(*attn_output.shape) > self.dropout
                attn_output = attn_output * mask_arr / (1 - self.dropout)

            x = x + attn_output
            x = self.ln1.forward(x)
            
            self.cache_after_ln1 = x.copy()

            ff_output = self.ffn.forward(x)

            if training and self.dropout > 0:
                mask_arr = np.random.rand(*ff_output.shape) > self.dropout
                ff_output = ff_output * mask_arr / (1 - self.dropout)

            x = x + ff_output
            x = self.ln2.forward(x)

            return x

    return RoPETransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout)


class ALiBiPositionalEncoding:
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    Alternative to RoPE that doesn't modify attention computation.
    """
    
    def __init__(self, num_heads, max_seq_length=128):
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        self._build_slopes()
    
    def _build_slopes(self):
        """Build attention bias slopes for each head."""
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(np.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]
            
            if np.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** np.floor(np.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
        
        slopes = get_slopes(self.num_heads)
        self.slopes = np.array(slopes).reshape(1, -1, 1, 1)
    
    def forward(self, seq_len):
        """Generate ALiBi attention mask."""
        positions = np.arange(seq_len).reshape(1, 1, -1, 1)
        relative_positions = positions - positions.transpose(0, 1, 3, 2)
        
        alibi = self.slopes * relative_positions
        
        mask = alibi.squeeze(0)
        
        return mask