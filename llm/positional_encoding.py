"""
Positional Encoding Module
Adds position information to token embeddings so the model knows
the order of tokens in a sequence.
Uses sine and cosine functions for different frequencies.
"""

import numpy as np


def get_positional_encoding(max_seq_length, d_model):
    """
    Create positional encoding matrix using sine and cosine functions.

    The formula uses different frequencies:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_length: Maximum sequence length to support
        d_model: Dimension of the model embeddings

    Returns:
        pe: Positional encoding matrix (max_seq_length x d_model)
    """
    pe = np.zeros((max_seq_length, d_model))

    # Create position indices
    position = np.arange(max_seq_length).reshape(-1, 1)

    # Create dimension indices
    dimension = np.arange(d_model).reshape(1, -1)

    # Calculate the division term for different frequencies
    # This creates different frequencies for different dimensions
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # Apply sine to even dimensions
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd dimensions
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def add_positional_encoding(embeddings, positional_encoding):
    """
    Add positional encoding to embeddings.

    Args:
        embeddings: Token embeddings (seq_len x d_model)
        positional_encoding: Precomputed positional encoding matrix

    Returns:
        encoded: Embeddings with positional information added
    """
    seq_len = embeddings.shape[0]
    encoded = embeddings + positional_encoding[:seq_len]
    return encoded


class PositionalEncoding:
    """
    Positional encoding layer.
    """

    def __init__(self, max_seq_length, d_model):
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.pe = get_positional_encoding(max_seq_length, d_model)

    def forward(self, embeddings):
        """
        Add positional encoding to embeddings.

        Args:
            embeddings: Token embeddings (batch_seq_len x d_model)

        Returns:
            positioned_embeddings: Embeddings with position info
        """
        return add_positional_encoding(embeddings, self.pe)

    def backward(self, grad_output):
        """
        Backward pass: positional encoding is not learnable, pass gradients through.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            grad_input: Gradient w.r.t. input (same as output since PE is fixed)
        """
        return grad_output

    def get_pe(self):
        """Get the positional encoding matrix."""
        return self.pe


def positional_encoding_manual(sequence_length, d_model):
    """
    Manual implementation of positional encoding computation.
    Useful for understanding what's happening step by step.

    Args:
        sequence_length: Length of the sequence
        d_model: Model dimension

    Returns:
        pe: Positional encoding matrix
    """
    pe = np.zeros((sequence_length, d_model))

    for pos in range(sequence_length):
        for i in range(0, d_model, 2):
            # Compute frequency for this dimension
            frequency = 1.0 / np.power(10000.0, (2.0 * i) / d_model)

            # Apply sine to even dimension
            pe[pos, i] = np.sin(pos * frequency)

            # Apply cosine to odd dimension
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos * frequency)

    return pe


class PositionalEncodingManual:
    """
    Manual implementation (slower but easier to understand).
    """

    def __init__(self, max_seq_length, d_model):
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.pe = positional_encoding_manual(max_seq_length, d_model)

    def forward(self, embeddings):
        """Add positional encoding."""
        seq_len = embeddings.shape[0]
        return embeddings + self.pe[:seq_len]