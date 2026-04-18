"""
QKV (Query, Key, Value) Module
Creates three linear projections from the input embeddings:
- Query: What we're looking for
- Key: What we're comparing against
- Value: What we want to extract if there's a match
These are the core components of scaled dot-product attention.
"""

import numpy as np


def init_qkv_weights(input_dim, num_heads, use_bias=True):
    """
    Initialize weights for Q, K, V projections.

    Each projection is a linear transformation: y = Wx + b
    For multi-head attention, we split the embedding into num_heads.

    Args:
        input_dim: Input dimension (d_model)
        num_heads: Number of attention heads
        use_bias: Whether to include bias terms

    Returns:
        weights: Dictionary with W_q, W_k, W_v, b_q, b_k, b_v
    """
    head_dim = input_dim // num_heads

    if head_dim * num_heads != input_dim:
        raise ValueError(f"d_model ({input_dim}) must be divisible by num_heads ({num_heads})")

    scale = np.sqrt(1.0 / input_dim)

    weights = {
        'W_q': np.random.randn(input_dim, input_dim) * scale,
        'W_k': np.random.randn(input_dim, input_dim) * scale,
        'W_v': np.random.randn(input_dim, input_dim) * scale,
    }

    if use_bias:
        weights['b_q'] = np.zeros(input_dim)
        weights['b_k'] = np.zeros(input_dim)
        weights['b_v'] = np.zeros(input_dim)
    else:
        weights['b_q'] = None
        weights['b_k'] = None
        weights['b_v'] = None

    return weights


def linear_projection(x, W, b=None):
    """
    Apply linear transformation: y = x @ W + b

    Args:
        x: Input matrix (batch, seq_len, input_dim) or (seq_len, input_dim)
        W: Weight matrix (input_dim, output_dim)
        b: Bias vector (output_dim) or None

    Returns:
        y: Output matrix with projected dimensions
    """
    if b is not None:
        return np.matmul(x, W) + b
    else:
        return np.matmul(x, W)


def compute_qkv(x, weights):
    """
    Compute Query, Key, Value projections from input.

    Args:
        x: Input embeddings (batch, seq_len, d_model) or (seq_len, d_model)
        weights: Dictionary with W_q, W_k, W_v, b_q, b_k, b_v

    Returns:
        q: Query projections
        k: Key projections
        v: Value projections
    """
    q = linear_projection(x, weights['W_q'], weights['b_q'])
    k = linear_projection(x, weights['W_k'], weights['b_k'])
    v = linear_projection(x, weights['W_v'], weights['b_v'])

    return q, k, v


def split_into_heads(q, k, v, num_heads):
    """
    Split Q, K, V matrices into multiple heads.

    For each head, we get a portion of the embedding dimension.

    Args:
        q: Query matrix (seq_len, d_model) or (batch, seq_len, d_model)
        k: Key matrix (seq_len, d_model) or (batch, seq_len, d_model)
        v: Value matrix (seq_len, d_model) or (batch, seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        q_heads: List of head matrices
        k_heads: List of head matrices
        v_heads: List of head matrices
    """
    # Handle both 2D and 3D inputs
    if len(q.shape) == 2:
        batch_size = 1
        seq_len, d_model = q.shape
        q = q.reshape(1, seq_len, d_model)
        k = k.reshape(1, seq_len, d_model)
        v = v.reshape(1, seq_len, d_model)
    else:
        batch_size, seq_len, d_model = q.shape

    head_dim = d_model // num_heads

    # Reshape: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
    q_heads = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_heads = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v_heads = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    return q_heads, k_heads, v_heads


def combine_heads(q_heads, k_heads, v_heads):
    """
    Combine head outputs back into a single matrix.

    Args:
        q_heads: Query heads (batch, num_heads, seq_len, head_dim)
        k_heads: Key heads (batch, num_heads, seq_len, head_dim)
        v_heads: Value heads (batch, num_heads, seq_len, head_dim)

    Returns:
        q: Combined query (batch, seq_len, d_model)
        k: Combined key (batch, seq_len, d_model)
        v: Combined value (batch, seq_len, d_model)
    """
    batch_size, num_heads, seq_len, head_dim = q_heads.shape
    d_model = num_heads * head_dim

    # Transpose: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
    q = q_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    k = k_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    v = v_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return q, k, v


class QKVProjection:
    """
    QKV projection layer for attention.
    """

    def __init__(self, d_model, num_heads, use_bias=True):
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.weights = init_qkv_weights(d_model, num_heads, use_bias)

    def forward(self, x):
        """
        Compute Q, K, V projections.

        Args:
            x: Input embeddings (batch, seq_len, d_model) or (seq_len, d_model)

        Returns:
            q, k, v: Projected matrices
        """
        # Ensure 3D input
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)

        return compute_qkv(x, self.weights)

    def get_weights(self):
        """Get QKV weights."""
        return self.weights

    def set_weights(self, weights):
        """Set QKV weights."""
        self.weights = weights


def init_separate_qkv(num_heads, head_dim, use_bias=True):
    """
    Initialize separate Q, K, V weights for each head.
    This is an alternative implementation where each head has its own weights.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        use_bias: Whether to include bias terms

    Returns:
        weights: Dictionary of weight matrices
    """
    weights = {
        'W_q': [],
        'W_k': [],
        'W_v': [],
    }

    scale = np.sqrt(1.0 / (num_heads * head_dim))

    for _ in range(num_heads):
        weights['W_q'].append(np.random.randn(head_dim, head_dim) * scale)
        weights['W_k'].append(np.random.randn(head_dim, head_dim) * scale)
        weights['W_v'].append(np.random.randn(head_dim, head_dim) * scale)

    if use_bias:
        weights['b_q'] = [np.zeros(head_dim) for _ in range(num_heads)]
        weights['b_k'] = [np.zeros(head_dim) for _ in range(num_heads)]
        weights['b_v'] = [np.zeros(head_dim) for _ in range(num_heads)]
    else:
        weights['b_q'] = [None] * num_heads
        weights['b_k'] = [None] * num_heads
        weights['b_v'] = [None] * num_heads

    return weights


def compute_qkv_per_head(x, weights, num_heads):
    """
    Compute Q, K, V for each head separately.

    Args:
        x: Input (batch, seq_len, num_heads * head_dim)
        weights: Per-head weights
        num_heads: Number of heads

    Returns:
        q_all, k_all, v_all: Combined projections
    """
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q_all = []
    k_all = []
    v_all = []

    # Process each head
    for h in range(num_heads):
        # Extract this head's portion of the input
        x_h = x[:, :, h*head_dim:(h+1)*head_dim]

        # Apply weights
        q_h = np.matmul(x_h, weights['W_q'][h])
        k_h = np.matmul(x_h, weights['W_k'][h])
        v_h = np.matmul(x_h, weights['W_v'][h])

        if weights['b_q'][h] is not None:
            q_h += weights['b_q'][h]
            k_h += weights['b_k'][h]
            v_h += weights['b_v'][h]

        q_all.append(q_h)
        k_all.append(k_h)
        v_all.append(v_h)

    # Stack heads: (batch, num_heads, seq_len, head_dim)
    q_all = np.stack(q_all, axis=1)
    k_all = np.stack(k_all, axis=1)
    v_all = np.stack(v_all, axis=1)

    return q_all, k_all, v_all