"""
linear.py — Linear layer and Feed-Forward Network (FFN).
FFN(x) = gelu(x @ W1 + b1) @ W2 + b2

C++ port:
  - Two GEMM calls with GELU in between
  - Weights stored as float* with row-major layout
  - bias_add fused into GEMM epilogue
"""

import numpy as np
from optimized_llm.tensor_ops import gelu, gelu_derivative
from optimized_llm.config import ModelConfig


def init_linear_weights(in_features: int, out_features: int, bias: bool = True) -> dict:
    """Xavier init for linear layer."""
    scale = np.sqrt(2.0 / (in_features + out_features))
    W = np.random.randn(in_features, out_features).astype(np.float32) * scale
    weights = {'W': W}
    if bias:
        weights['b'] = np.zeros(out_features, dtype=np.float32)
    return weights


def linear_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """y = x @ W + b"""
    out = np.matmul(x, weights['W'])
    if weights.get('b') is not None:
        out = out + weights['b']
    return out


class Linear:
    """Simple linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weights = init_linear_weights(in_features, out_features, bias)
        self.last_input: np.ndarray = None
        self.last_grad_W: np.ndarray = None  # stored for optimizer

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return linear_forward(x, self.weights)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients, store them, and return grad_input.
        C++: dW = x^T @ dout, dx = dout @ W^T
        """
        W = self.weights['W']
        x_2d = self.last_input.reshape(-1, W.shape[0])
        g_2d = grad_output.reshape(-1, W.shape[1])

        self.last_grad_W = np.matmul(x_2d.T, g_2d)
        if self.weights.get('b') is not None:
            self.last_grad_b = np.sum(g_2d, axis=0)

        return np.matmul(g_2d, W.T).reshape(self.last_input.shape)


class FeedForward:
    """
    Two-layer FFN with GELU activation.
    FFN(x) = gelu(x @ W1 + b1) @ W2 + b2
    """

    def __init__(self, cfg: ModelConfig):
        self.W1 = init_linear_weights(cfg.d_model, cfg.d_ff, bias=True)
        self.W2 = init_linear_weights(cfg.d_ff, cfg.d_model, bias=True)
        self.last_input: np.ndarray = None
        self.last_hidden: np.ndarray = None  # after GELU, before W2
        self.last_grad_W1: np.ndarray = None
        self.last_grad_W2: np.ndarray = None
        self.last_grad_b1: np.ndarray = None
        self.last_grad_b2: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq, d_model) → (batch, seq, d_model)
        """
        self.last_input = x
        hidden = linear_forward(x, self.W1)   # (batch, seq, d_ff)
        self.last_pre_act = hidden             # before GELU, for backward
        hidden_act = gelu(hidden)              # (batch, seq, d_ff)
        self.last_hidden = hidden_act
        output = linear_forward(hidden_act, self.W2)  # (batch, seq, d_model)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backprop through FFN with gradient storage.
        C++: chain rule through two linear layers and GELU.
        """
        W2 = self.W2['W']
        x = self.last_input
        hidden = self.last_pre_act
        grad_act = gelu_derivative(hidden)

        # Grad through W2: dL/dH = dL/dOut @ W2^T
        g_hidden = np.matmul(grad_output, W2.T)   # (batch, seq, d_ff)
        g_hidden = g_hidden * grad_act

        # Store W2 gradient: dL/dW2 = H^T @ dL/dOut
        h_2d = self.last_hidden.reshape(-1, W2.shape[0])
        g_out_2d = grad_output.reshape(-1, W2.shape[1])
        self.last_grad_W2 = np.matmul(h_2d.T, g_out_2d)
        if self.W2['b'] is not None:
            self.last_grad_b2 = np.sum(g_out_2d, axis=0)

        # Grad through W1: dL/dX = dL/dH @ W1^T
        g_x = np.matmul(g_hidden, self.W1['W'].T)  # (batch, seq, d_model)

        # Store W1 gradient: dL/dW1 = X^T @ dL/dH
        x_2d = x.reshape(-1, self.W1['W'].shape[0])
        g_h_2d = g_hidden.reshape(-1, self.W1['W'].shape[1])
        self.last_grad_W1 = np.matmul(x_2d.T, g_h_2d)
        if self.W1['b'] is not None:
            self.last_grad_b1 = np.sum(g_h_2d, axis=0)

        return g_x
