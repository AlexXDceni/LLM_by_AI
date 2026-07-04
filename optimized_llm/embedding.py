"""
embedding.py — Token embedding layer.
Maps integer token IDs → dense vectors.
C++: this is a simple table lookup `float* emb = &table[token_id * dim];`.
"""

import numpy as np
from optimized_llm.config import ModelConfig


class TokenEmbedding:
    """
    Token embedding layer.
    Forward:  token_ids (batch, seq) → embeddings (batch, seq, d_model)
    Backward: gradient w.r.t. embeddings → gradient w.r.t. embedding matrix
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        scale = 0.1
        self.W: np.ndarray = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * scale
        self.last_input: np.ndarray = None  # (batch, seq) for backward
        self.last_grad: np.ndarray = None   # (batch, seq, d_model) stored by training loop

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Embedding lookup via indexing.
        C++: for i in 0..batch*seq: out[i*d_model + j] = table[tokens[i]*d_model + j]
        """
        self.last_input = np.asarray(token_ids, dtype=np.int32)
        return self.W[self.last_input]  # (batch, seq, d_model)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Accumulate gradients into embedding matrix using np.add.at.
        C++: atomicAdd or serial loop.
        grad_output: (batch, seq, d_model)
        returns: grad_W (vocab_size, d_model)
        """
        flat_tokens = self.last_input.reshape(-1)
        flat_grad = grad_output.reshape(-1, self.cfg.d_model)

        self.last_grad = grad_output

        grad_W = np.zeros_like(self.W)
        np.add.at(grad_W, flat_tokens, flat_grad)
        return grad_W
