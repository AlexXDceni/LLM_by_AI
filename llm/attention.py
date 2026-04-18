"""
Attention Module
Implements scaled dot-product attention, the core mechanism of transformers.
This allows the model to focus on different parts of the input when processing.
"""

import numpy as np
from llm.softmax import scaled_softmax, mask_softmax
from llm.qkv import compute_qkv, split_into_heads


def scaled_dot_product_attention(q, k, v, mask=None, scale=None):
    """
    Compute scaled dot-product attention.

    The attention score measures how much each key "matches" the query.
    We use these scores to weight the values.

    Steps:
    1. Compute similarity: similarity = Q @ K^T / sqrt(d_k)
    2. Apply mask (optional)
    3. Convert scores to probabilities with softmax
    4. Weight values: output = probabilities @ V

    Args:
        q: Query tensor (batch, num_heads, seq_len_q, head_dim)
        k: Key tensor (batch, num_heads, seq_len_k, head_dim)
        v: Value tensor (batch, num_heads, seq_len_v, head_dim)
        mask: Optional mask (batch, num_heads, seq_len_q, seq_len_k)
        scale: Scale factor (usually sqrt(head_dim))

    Returns:
        output: Attention-weighted values
        attention_weights: Attention probabilities
    """
    # Use head_dim from Q if scale not provided
    if scale is None:
        scale = np.sqrt(q.shape[-1])

    # Compute dot products: (batch, num_heads, seq_len_q, seq_len_k)
    # Q @ K^T
    attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2))

    # Scale
    attention_scores = attention_scores / scale

    # Apply mask if provided
    if mask is not None:
        attention_scores = np.where(mask == False, -1e9, attention_scores)

    # Convert to probabilities
    attention_weights = scaled_softmax(attention_scores, 1.0, axis=-1)

    # Weight values: (batch, num_heads, seq_len_q, head_dim)
    output = np.matmul(attention_weights, v)

    return output, attention_weights


def self_attention(x, qkv_weights, num_heads, mask=None):
    """
    Self-attention: all inputs attend to all other inputs.

    Args:
        x: Input embeddings (seq_len, d_model) or (batch, seq_len, d_model)
        qkv_weights: Dictionary with Q, K, V weights
        num_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        output: Self-attention output
        attention_weights: Attention probabilities
    """
    from llm.qkv import compute_qkv, split_into_heads, combine_heads

    # Ensure 3D input
    if len(x.shape) == 2:
        x = x.reshape(1, *x.shape)

    # Compute Q, K, V
    q, k, v = compute_qkv(x, qkv_weights)

    # Split into heads
    q_heads, k_heads, v_heads = split_into_heads(q, k, v, num_heads)

    # Compute attention
    output_heads, attention_weights = scaled_dot_product_attention(
        q_heads, k_heads, v_heads, mask
    )

    # Combine heads back together
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    d_model = x.shape[2]
    head_dim = d_model // num_heads

    output = output_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return output, attention_weights


def causal_mask(seq_len):
    """
    Create a causal mask for autoregressive generation.
    Each position can only attend to previous positions.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Boolean mask (True = allowed, False = masked)
    """
    # Create lower triangular mask
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


def padding_mask(token_ids, pad_token_id=0):
    """
    Create a padding mask to ignore padding tokens.

    Args:
        token_ids: Token IDs (batch, seq_len)
        pad_token_id: ID of the padding token

    Returns:
        mask: Boolean mask (True = valid, False = padding)
    """
    return token_ids != pad_token_id


class MultiHeadAttention:
    """
    Multi-head attention layer.
    """

    def __init__(self, d_model, num_heads, use_bias=True, dropout=0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = np.sqrt(self.head_dim)
        self.dropout = dropout

        from llm.qkv import init_qkv_weights
        self.qkv_weights = init_qkv_weights(d_model, num_heads, use_bias)
        self.last_qkv = None

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, d_model) or (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attention output
            attention_weights: Attention probabilities
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)

        self.last_x = x.copy()
        
        output, attn_weights = self_attention(x, self.qkv_weights, self.num_heads, mask)
        
        if self.dropout > 0:
            mask_dropout = np.random.rand(*output.shape) > self.dropout
            output = output * mask_dropout / (1 - self.dropout)
        
        return output, attn_weights

    def backward(self, grad_output):
        """
        Backward pass for multi-head attention.
        """
        x = self.last_x
        batch_size, seq_len, d_model = x.shape
        
        q, k, v = compute_qkv(x, self.qkv_weights)
        q_heads, k_heads, v_heads = split_into_heads(q, k, v, self.num_heads)
        
        head_dim = self.head_dim
        
        attention_scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / self.scale
        attention_weights = scaled_softmax(attention_scores, 1.0, axis=-1)
        
        output = np.matmul(attention_weights, v_heads)
        
        grad_output_reshaped = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        grad_v = np.matmul(attention_weights.transpose(0, 1, 3, 2).reshape(batch_size * seq_len, seq_len * self.num_heads),
                          v_heads.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, head_dim))
        grad_v = grad_v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        
        grad_q = np.zeros_like(q_heads)
        grad_k = np.zeros_like(k_heads)
        
        grad_input = np.matmul(grad_output_reshaped, self.qkv_weights['W_q'].T)
        
        return grad_input

    def get_weights(self):
        """Get QKV weights."""
        return self.qkv_weights

    def set_weights(self, weights):
        """Set QKV weights."""
        self.qkv_weights = weights


class SelfAttention:
    """
    Simple self-attention wrapper for single sequences.
    """

    def __init__(self, d_model, num_heads):
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        """Apply self-attention."""
        return self.attention.forward(x, mask)


def multihead_attention_manual(q, k, v, mask=None):
    """
    Manual implementation of multi-head attention.
    Step-by-step for understanding purposes.

    Args:
        q: Query
        k: Key
        v: Value
        mask: Optional mask

    Returns:
        output: Attention output
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Step 1: Compute attention scores for each head
    scores = np.zeros((batch_size, num_heads, seq_len, seq_len))

    for h in range(num_heads):
        q_h = q[:, h, :, :]  # (batch, seq_len, head_dim)
        k_h = k[:, h, :, :]  # (batch, seq_len, head_dim)

        # Dot product: (batch, seq_len, seq_len)
        scores[:, h, :, :] = np.matmul(q_h, k_h.transpose(0, 2, 1)) / np.sqrt(head_dim)

    # Step 2: Apply softmax
    if mask is not None:
        scores = np.where(mask == False, -1e9, scores)

    weights = np.array([scaled_softmax(scores[:, h], axis=-1) for h in range(num_heads)])
    weights = weights.transpose(1, 0, 2, 3)

    # Step 3: Weight values
    output = np.zeros_like(q)

    for h in range(num_heads):
        v_h = v[:, h, :, :]  # (batch, seq_len, head_dim)
        output[:, h, :, :] = np.matmul(weights[:, h], v_h)

    return output, weights


def cross_attention(q, kv, num_heads, mask=None):
    """
    Cross-attention: query comes from one sequence, keys/values from another.
    Used in decoder of encoder-decoder transformers.

    Args:
        q: Query sequence (batch, seq_len_q, d_model)
        kv: Key-Value sequence (batch, seq_len_kv, d_model)
        num_heads: Number of attention heads
        mask: Optional mask

    Returns:
        output: Cross-attention output
    """
    from llm.qkv import compute_qkv, split_into_heads

    # Ensure 3D inputs
    if len(q.shape) == 2:
        q = q.reshape(1, *q.shape)
    if len(kv.shape) == 2:
        kv = kv.reshape(1, *kv.shape)

    # For cross attention, use separate projections
    from llm.qkv import init_qkv_weights
    qkv_w = init_qkv_weights(q.shape[-1], num_heads, use_bias=True)

    # Get Q from first sequence
    q, _, _ = compute_qkv(q, qkv_w)

    # Get K, V from second sequence
    _, k, v = compute_qkv(kv, qkv_w)

    # Split into heads
    q_heads, k_heads, v_heads = split_into_heads(q, k, v, num_heads)

    # Compute attention
    output_heads, _ = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

    # Combine heads
    output = output_heads.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], -1)

    return output