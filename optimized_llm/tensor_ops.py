"""
tensor_ops.py — Fused elementary operations.
Each function is: stateless, typed, C++-portable.
No Python loops, minimal temporary allocations.
"""

import numpy as np


def fused_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax fused into one pass.
    C++: subtract max, exp, sum, divide — all in one loop.
    x: (..., dim) or (batch, heads, seq, seq)
    """
    x_max = np.max(x, axis=axis, keepdims=True)          # reduce
    x_shifted = x - x_max                                 # broadcast
    exp_x = np.exp(x_shifted)                             # element-wise
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)     # reduce
    return exp_x / sum_exp                                # broadcast ÷


def fused_softmax_masked(x: np.ndarray, mask: np.ndarray,
                         mask_val: float = -1e9) -> np.ndarray:
    """
    Softmax with optional mask fused.
    C++: `if (mask) scores[i] = mask_val;` before softmax kernel.
    """
    x = np.where(mask, mask_val, x)
    return fused_softmax(x)


def fused_layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-8) -> np.ndarray:
    """
    LayerNorm fused: mean + var computed in one pass.
    C++: single loop computing sum(x) and sum(x²) simultaneously.
    x: (batch, seq, d_model)
    gamma, beta: (d_model,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def fused_rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    RMSNorm — simpler than LayerNorm, no centering.
    C++: one loop for sum(x²), then divide-and-scale.
    x: (batch, seq, d_model) or (..., d_model)
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (tanh approximation).
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    """
    coeff = np.sqrt(2.0 / np.pi)
    inner = coeff * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU for backprop."""
    coeff = np.sqrt(2.0 / np.pi)
    c = 0.044715
    t = np.tanh(coeff * (x + c * x ** 3))
    dtanh = coeff * (1.0 + 3.0 * c * x ** 2) * (1.0 - t ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * dtanh


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss: stable, fused softmax internally.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)  — integer token IDs
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = logits_shifted - np.log(sum_exp + 1e-10)

    losses = -log_probs[np.arange(len(targets_flat)), targets_flat]
    return float(np.mean(losses))


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Gradient of cross-entropy w.r.t. logits: (probs - one_hot) / batch.
    Returns: (batch, seq_len, vocab_size)
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_flat - logits_max)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

    batch_size = logits.shape[0] * logits.shape[1]
    grad = probs.copy()
    grad[np.arange(len(targets_flat)), targets_flat] -= 1.0
    return grad.reshape(logits.shape) / batch_size
