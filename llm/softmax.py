"""
Softmax Module
Applies softmax function to convert raw scores into probability distributions.
Used in attention mechanism to normalize attention scores.
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Apply softmax function along specified axis.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    This creates a probability distribution where all values sum to 1.

    Args:
        x: Input array of any shape
        axis: Axis along which to apply softmax

    Returns:
        softmax_output: Softmax probabilities
    """
    # Numerical stability: subtract max to prevent overflow
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)

    # Sum exponentials
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)

    # Divide by sum
    softmax_output = x_exp / x_sum

    return softmax_output


def scaled_softmax(x, scale, axis=-1):
    """
    Apply scaled softmax: softmax(x / scale)

    Args:
        x: Input scores
        scale: Scale factor (usually sqrt(d_k) for attention)
        axis: Axis along which to apply softmax

    Returns:
        scaled_softmax_output: Scaled softmax probabilities
    """
    return softmax(x / scale, axis=axis)


def mask_softmax(x, mask, axis=-1, mask_value=-1e9):
    """
    Apply softmax with masking.

    Sets masked positions to a very negative value before softmax,
    so they become effectively zero after normalization.

    Args:
        x: Input scores (batch, num_heads, seq_len, seq_len)
        mask: Boolean mask (True = masked/ignore)
        axis: Axis along which to apply softmax
        mask_value: Value to use for masking

    Returns:
        masked_softmax: Softmax with masked positions set to ~0
    """
    # Apply mask by setting masked positions to very negative
    x_masked = np.where(mask, mask_value, x)

    # Apply softmax
    return softmax(x_masked, axis=axis)


def softmax_backward(x, grad_output, axis=-1):
    """
    Backward pass for softmax.

    Computing the gradient of softmax is tricky because softmax
    itself depends on all input values.

    dL/dx_i = sum_j(dL/dy_j * y_j * (delta_ij - y_i))

    where y = softmax(x)

    Args:
        x: Original input to softmax (needed for forward computation)
        grad_output: Gradient with respect to softmax output
        axis: Axis along which softmax was applied

    Returns:
        grad_input: Gradient with respect to input
    """
    # Compute softmax output
    y = softmax(x, axis=axis)

    # Jacobian: diag(y) - outer(y, y)
    # For numerical stability, we use a simpler approach

    # Sum gradients weighted by output
    grad_input = y * np.sum(grad_output, axis=axis, keepdims=True)

    # Subtract contribution from this element
    # dL/dx_i = sum_j(dL/dy_j * y_j) * y_i - dL/dy_i * y_i
    temp = np.sum(grad_output * y, axis=axis, keepdims=True)
    grad_input = grad_input - y * temp

    return grad_input


class Softmax:
    """
    Softmax layer.
    """

    def __init__(self, axis=-1):
        self.axis = axis
        self.output = None

    def forward(self, x):
        """Apply softmax."""
        self.output = softmax(x, self.axis)
        return self.output

    def backward(self, grad_output):
        """Backward pass."""
        if self.output is None:
            raise ValueError("Must call forward before backward")

        # Need input for backward - this is a simplified version
        # In practice, you'd want to store input during forward
        input_for_grad = np.log(self.output + 1e-9)
        return softmax_backward(input_for_grad, grad_output, self.axis)


class ScaledSoftmax:
    """
    Scaled softmax for attention.
    """

    def __init__(self, scale_factor, axis=-1):
        self.scale_factor = scale_factor
        self.axis = axis

    def forward(self, x):
        """Apply scaled softmax."""
        return scaled_softmax(x, self.scale_factor, self.axis)


class MaskedSoftmax:
    """
    Softmax with mask support for attention.
    """

    def __init__(self, mask_value=-1e9, axis=-1):
        self.mask_value = mask_value
        self.axis = axis

    def forward(self, x, mask):
        """Apply masked softmax."""
        return mask_softmax(x, mask, self.axis, self.mask_value)


def stable_softmax(x, axis=-1):
    """
    Numerically stable softmax implementation.
    Subtracts max before exponentiation to prevent overflow.

    Args:
        x: Input array
        axis: Axis along which to apply softmax

    Returns:
        Result: Softmax output
    """
    # For 1D arrays
    if len(x.shape) == 1:
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    # For multi-dimensional
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """
    Log softmax: log(softmax(x))
    Useful for numerical stability in loss functions.

    Args:
        x: Input array
        axis: Axis along which to apply

    Returns:
        Log softmax output
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return x - x_max - log_sum