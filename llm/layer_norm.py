"""
Layer Normalization Module
Normalizes activations across the feature dimension.
Helps stabilize training and speeds up convergence.
Unlike batch norm, it normalizes each sample independently (no batch dependency).
"""

import numpy as np


def compute_mean(x, axis=-1, keepdims=True):
    """
    Compute mean along specified axis.

    Args:
        x: Input tensor
        axis: Axis to compute mean along
        keepdims: Whether to keep dimensions

    Returns:
        mean: Mean value(s)
    """
    return np.mean(x, axis=axis, keepdims=keepdims)


def compute_variance(x, axis=-1, keepdims=True, epsilon=1e-8):
    """
    Compute variance along specified axis.

    variance = mean((x - mean)^2)

    Args:
        x: Input tensor
        axis: Axis to compute variance along
        keepdims: Whether to keep dimensions
        epsilon: Small value to prevent division by zero

    Returns:
        variance: Variance value(s)
    """
    mean = compute_mean(x, axis, keepdims)
    variance = np.mean((x - mean) ** 2, axis=axis, keepdims=keepdims)
    return variance + epsilon


def layer_norm(x, gamma, beta, epsilon=1e-8):
    """
    Layer normalization.

    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    Normalizes across the last dimension (feature dimension).

    Args:
        x: Input tensor (any shape, last dim is features)
        gamma: Scale parameter (feature_dim,)
        beta: Shift parameter (feature_dim,)
        epsilon: Small value to prevent division by zero

    Returns:
        normalized: Layer-normalized output
    """
    # Compute mean and variance
    mean = compute_mean(x, axis=-1, keepdims=True)
    variance = compute_variance(x, axis=-1, keepdims=True, epsilon=epsilon)

    # Normalize
    x_centered = x - mean
    std = np.sqrt(variance)
    norm_output = x_centered / std

    # Scale and shift
    norm_output = gamma * norm_output + beta

    return norm_output


def layer_norm_forward(x, gamma, beta, epsilon=1e-8):
    """
    Forward pass for layer normalization.

    Returns normalized output and cache for backward pass.
    """
    mean = compute_mean(x, axis=-1, keepdims=True)
    variance = compute_variance(x, axis=-1, keepdims=True, epsilon=epsilon)

    std = np.sqrt(variance)
    x_centered = x - mean
    normalized = x_centered / std

    output = gamma * normalized + beta

    # Cache for backward
    cache = {
        'x': x,
        'mean': mean,
        'variance': variance,
        'std': std,
        'normalized': normalized,
        'gamma': gamma,
        'beta': beta
    }

    return output, cache


def layer_norm_backward(cache, grad_output):
    """
    Backward pass for layer normalization.
    Simplified version - pass gradient through.
    """
    x = cache['x']
    gamma = cache['gamma']
    
    # Simplified backward: just pass gradient through scaled by gamma
    grad_input = grad_output * gamma
    
    # Compute grad_gamma and grad_beta
    if len(x.shape) == 3:
        # (batch, seq_len, feature_dim)
        grad_gamma = np.sum(grad_output * (x - cache['mean']) / cache['std'], axis=(0, 1))
        grad_beta = np.sum(grad_output, axis=(0, 1))
    else:
        # (batch, feature_dim)
        grad_gamma = np.sum(grad_output * (x - cache['mean']) / cache['std'], axis=0)
        grad_beta = np.sum(grad_output, axis=0)
    
    return grad_gamma, grad_beta, grad_input


class LayerNorm:
    """
    Layer normalization layer.
    """

    def __init__(self, normalized_shape, epsilon=1e-8):
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape

        self.epsilon = epsilon

        feature_dim = self.normalized_shape[-1]
        self.gamma = np.ones(feature_dim)
        self.beta = np.zeros(feature_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, feature_dim) or (batch, feature_dim)

        Returns:
            normalized: Layer-normalized output
        """
        output, self.cache = layer_norm_forward(x, self.gamma, self.beta, self.epsilon)
        return output

    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output
        
        Returns:
            grad_input: Gradient w.r.t. input
        """
        grad_gamma, grad_beta, grad_input = layer_norm_backward(self.cache, grad_output)
        
        self.gamma -= 0.001 * grad_gamma
        self.beta -= 0.001 * grad_beta
        
        return grad_input

    def get_parameters(self):
        """Get gamma and beta."""
        return {'gamma': self.gamma, 'beta': self.beta}

    def set_parameters(self, params):
        """Set gamma and beta."""
        self.gamma = params['gamma']
        self.beta = params['beta']


class BatchNorm:
    """
    Batch normalization (alternative - less common in transformers).

    Normalizes across batch and sequence dimensions.
    """

    def __init__(self, normalized_shape, momentum=0.9, epsilon=1e-8):
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape

        self.momentum = momentum
        self.epsilon = epsilon

        feature_dim = self.normalized_shape[-1]
        self.gamma = np.ones(feature_dim)
        self.beta = np.zeros(feature_dim)

        self.running_mean = np.zeros(feature_dim)
        self.running_var = np.ones(feature_dim)

    def forward(self, x, training=True):
        """
        Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            normalized: Batch-normalized output
        """
        if training:
            # Compute mean and variance
            mean = np.mean(x, axis=(0, 1), keepdims=True)
            variance = np.var(x, axis=(0, 1), keepdims=True)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance.squeeze()
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, 1, -1)
            variance = self.running_var.reshape(1, 1, -1)

        # Normalize
        std = np.sqrt(variance + self.epsilon)
        normalized = (x - mean) / std

        # Scale and shift
        return self.gamma * normalized + self.beta


def rms_norm(x, gamma, epsilon=1e-8):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Simpler than layer norm - only normalizes by RMS, no centering.

    RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)

    Args:
        x: Input tensor
        gamma: Scale parameter
        epsilon: Small value to prevent division by zero

    Returns:
        normalized: RMS-normalized output
    """
    # Compute RMS (root mean square)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + epsilon)

    # Normalize
    normalized = x / rms

    # Scale
    return gamma * normalized


class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, normalized_shape, epsilon=1e-8):
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape

        self.epsilon = epsilon
        feature_dim = self.normalized_shape[-1]
        self.gamma = np.ones(feature_dim)

    def forward(self, x):
        """Forward pass."""
        return rms_norm(x, self.gamma, self.epsilon)