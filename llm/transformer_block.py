"""
Transformer Block Module
A complete transformer block consists of:
1. Multi-head self-attention with residual connection
2. Add & LayerNorm
3. Feed-forward network with residual connection
4. Add & LayerNorm

This is the building block of the transformer architecture.
"""

import numpy as np
from llm.attention import MultiHeadAttention
from llm.linear import FeedForward, Linear
from llm.layer_norm import LayerNorm


def build_transformer_block(d_model, num_heads, d_ff=None, dropout=0.0):
    """
    Create all components needed for a transformer block.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout rate

    Returns:
        components: Dictionary with all layer components
    """
    if d_ff is None:
        d_ff = d_model * 4

    components = {
        'attention': MultiHeadAttention(d_model, num_heads, use_bias=True),
        'ffn': FeedForward(d_model, d_ff, activation='gelu'),
        'ln1': LayerNorm(d_model),
        'ln2': LayerNorm(d_model),
    }

    return components


def transformer_block_forward(x, components, mask=None, dropout=0.0, training=True):
    """
    Apply one transformer block.

    Args:
        x: Input tensor (batch, seq_len, d_model)
        components: Dictionary with block components
        mask: Optional attention mask
        dropout: Dropout rate
        training: Whether in training mode

    Returns:
        output: Transformer block output
    """
    # Save input for residual connection
    x_residual = x

    # Multi-head self-attention
    attention_output, _ = components['attention'].forward(x, mask)

    # Apply dropout during training
    if training and dropout > 0:
        mask_dropout = np.random.rand(*attention_output.shape) > dropout
        attention_output = attention_output * mask_dropout / (1 - dropout)

    # Add residual connection
    x = x_residual + attention_output

    # First layer norm
    x = components['ln1'].forward(x)

    # Save for residual connection
    x_residual = x

    # Feed-forward network
    ff_output = components['ffn'].forward(x)

    # Apply dropout
    if training and dropout > 0:
        mask_dropout = np.random.rand(*ff_output.shape) > dropout
        ff_output = ff_output * mask_dropout / (1 - dropout)

    # Add residual connection
    x = x_residual + ff_output

    # Second layer norm
    x = components['ln2'].forward(x)

    return x


class TransformerBlock:
    """
    Transformer block layer.
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.dropout = dropout

        self.attention = MultiHeadAttention(d_model, num_heads, use_bias=True)
        self.ffn = FeedForward(d_model, self.d_ff, activation='gelu')
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None, training=True):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            training: Training mode flag

        Returns:
            output: Transformer block output
        """
        self.cache_x = x.copy()
        
        attn_output, _ = self.attention.forward(x, mask)

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

    def backward(self, grad_output):
        """
        Backward pass through transformer block.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            grad_input: Gradient w.r.t. input
        """
        grad = self.ln2.backward(grad_output)
        
        grad_ffn = grad.copy()
        
        grad_x = grad.copy()
        
        grad = self.ffn.backward(grad_x)
        
        grad_attn = grad + self.cache_x
        grad = self.ln1.backward(grad_attn)
        
        grad_attn_out, _ = self.attention.backward(grad)
        
        grad_input = grad_attn_out + self.cache_x
        
        return grad_input

    def get_components(self):
        """Get all components."""
        return {
            'attention': self.attention,
            'ffn': self.ffn,
            'ln1': self.ln1,
            'ln2': self.ln2,
        }


class DecoderBlock:
    """
    Decoder block for autoregressive transformers.
    Similar to transformer block but with causal masking.
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x, mask=None, training=True):
        """Forward pass with causal mask."""
        seq_len = x.shape[1]

        # Create causal mask if not provided
        if mask is None:
            from attention import causal_mask
            causal = causal_mask(seq_len)
            # Expand to proper shape for attention
            mask = ~causal  # True means masked

        return self.transformer_block.forward(x, mask, training)


class EncoderBlock:
    """
    Encoder block for bidirectional transformers.
    No causal masking needed (uses full attention).
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout)

    def forward(self, x, mask=None, training=True):
        """Forward pass."""
        return self.transformer_block.forward(x, mask, training)


def build_transformer_stack(num_layers, d_model, num_heads, d_ff=None, dropout=0.0):
    """
    Build a stack of transformer blocks.

    Args:
        num_layers: Number of transformer blocks
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate

    Returns:
        blocks: List of transformer blocks
    """
    blocks = []

    for _ in range(num_layers):
        block = TransformerBlock(d_model, num_heads, d_ff, dropout)
        blocks.append(block)

    return blocks


def transformer_stack_forward(x, blocks, mask=None, training=True):
    """
    Apply stack of transformer blocks sequentially.

    Args:
        x: Input tensor
        blocks: List of transformer blocks
        mask: Optional attention mask
        training: Training mode

    Returns:
        output: Final output after all blocks
    """
    for block in blocks:
        x = block.forward(x, mask, training)

    return x


class TransformerStack:
    """
    Stack of transformer blocks with dropout.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=None, dropout=0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.dropout = dropout

        self.blocks = build_transformer_stack(num_layers, d_model, num_heads, self.d_ff, dropout)

    def forward(self, x, mask=None, training=True, return_cache=False, use_cache=False, kv_cache=None):
        """Forward through all blocks."""
        cache = {'block_inputs': []}
        new_cache = [] if use_cache else None
        
        for block in self.blocks:
            cache['block_inputs'].append(x.copy())
            
            if use_cache and kv_cache is not None:
                pass
            
            x = block.forward(x, mask, training)
            
            if use_cache:
                new_cache.append(x.copy())
        
        if return_cache:
            return x, cache
        return x if not use_cache else (x, new_cache)

    def backward(self, grad_output):
        """Backward through all blocks in reverse order."""
        grad = grad_output
        
        for i in range(self.num_layers - 1, -1, -1):
            block = self.blocks[i]
            input_x = block.cache_x
            grad = block.backward(grad)
        
        return grad

    def get_blocks(self):
        """Get all blocks."""
        return self.blocks


def init_weights(vocab_size, d_model, num_layers, num_heads):
    """
    Initialize all weights for the transformer.

    This creates the complete parameter set for a transformer model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads

    Returns:
        weights: Dictionary with all model weights
    """
    from embeddings import init_embedding_matrix

    weights = {
        'token_embedding': init_embedding_matrix(vocab_size, d_model),
    }

    # Add weights for each transformer block
    for i in range(num_layers):
        block_weights = {}

        # Attention weights
        block_weights['attention'] = weights['attention'].get_weights()
        block_weights['ffn'] = weights['ffn'].get_weights()
        block_weights['ln1'] = weights['ln1'].get_parameters()
        block_weights['ln2'] = weights['ln2'].get_parameters()

        weights[f'block_{i}'] = block_weights

    return weights