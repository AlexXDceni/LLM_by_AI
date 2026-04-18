"""
Embeddings Module
Converts integer token IDs into dense vector representations.
This is the first step in transforming text into meaningful representations.
"""

import numpy as np


def init_embedding_matrix(vocab_size, embedding_dim, scale=0.1):
    """
    Initialize embedding matrix with small random values.
    This matrix maps each token ID to a dense embedding vector.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        embedding_dim: Dimension of the embedding vectors
        scale: Scale factor for random initialization

    Returns:
        embedding_matrix: Shape (vocab_size, embedding_dim)
    """
    # Initialize with random values centered around 0
    embedding_matrix = np.random.randn(vocab_size, embedding_dim) * scale
    return embedding_matrix


def lookup_embeddings(token_ids, embedding_matrix):
    """
    Look up embeddings for given token IDs.
    This converts a sequence of token IDs into their embedding vectors.

    Args:
        token_ids: Array of token IDs (shape: sequence_length)
        embedding_matrix: Embedding matrix (shape: vocab_size x embedding_dim)

    Returns:
        embeddings: Embedded vectors (shape: sequence_length x embedding_dim)
    """
    # Ensure token_ids is an integer array
    token_ids = np.array(token_ids, dtype=int)

    # Look up each token's embedding
    embeddings = embedding_matrix[token_ids]

    return embeddings


def train_embeddings(token_ids, embedding_matrix, gradients, learning_rate):
    """
    Update embedding matrix based on gradients.
    This is the backward pass for embeddings.

    Args:
        token_ids: Array of token IDs used
        embedding_matrix: Current embedding matrix
        gradients: Gradient with respect to embeddings
        learning_rate: Learning rate for updates

    Returns:
        updated_embedding_matrix: Updated embedding matrix
    """
    token_ids = np.array(token_ids, dtype=int)

    # Update only the embeddings that were used
    for i, token_id in enumerate(token_ids):
        embedding_matrix[token_id] -= learning_rate * gradients[i]

    return embedding_matrix


class TokenEmbedding:
    """
    Token embedding layer.
    """

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = {'W': init_embedding_matrix(vocab_size, embedding_dim)}
        self.last_input = None

    def forward(self, token_ids):
        """
        Forward pass: convert token IDs to embeddings.
        """
        self.last_input = np.array(token_ids, dtype=int)
        return lookup_embeddings(token_ids, self.weights['W'])

    def backward(self, grad_output):
        """
        Backward pass: compute gradients w.r.t. embeddings.
        
        Args:
            grad_output: Gradient from next layer (batch, seq_len, emb_dim)
        
        Returns:
            grad_input: Gradient w.r.t. input (for upstream)
        """
        batch_size, seq_len, emb_dim = grad_output.shape
        
        grad_emb = np.zeros((self.vocab_size, emb_dim))
        
        input_tokens = self.last_input.reshape(-1)
        grad_output_flat = grad_output.reshape(-1, emb_dim)
        
        for i, token_id in enumerate(input_tokens):
            grad_emb[token_id] += grad_output_flat[i]
        
        grad_input = None
        
        return grad_emb

    def get_embeddings(self):
        """Get the embedding matrix."""
        return self.weights['W']

    def set_embeddings(self, embedding_matrix):
        """Set a custom embedding matrix."""
        self.weights['W'] = embedding_matrix


class Embedding:
    """
    Simple alias for TokenEmbedding.
    """

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = init_embedding_matrix(vocab_size, embedding_dim)

    def forward(self, token_ids):
        """Forward pass."""
        return lookup_embeddings(token_ids, self.embedding_matrix)

    def backward(self, token_ids, gradients, learning_rate):
        """Backward pass."""
        self.embedding_matrix = train_embeddings(
            token_ids, self.embedding_matrix, gradients, learning_rate
        )

    def get_weights(self):
        """Get embedding matrix."""
        return self.embedding_matrix

    def set_weights(self, embedding_matrix):
        """Set embedding matrix."""
        self.embedding_matrix = embedding_matrix