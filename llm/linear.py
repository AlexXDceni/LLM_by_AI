"""
Linear (Feed-Forward) Module
Implements the position-wise feed-forward network in transformers.
Typically has two linear layers with a GELU activation in between.
Adds non-linearity and allows the model to transform features.
"""

import numpy as np


def init_linear_weights(input_dim, output_dim, bias=True):
    """
    Initialize weights for a linear layer.

    y = x @ W^T + b

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        bias: Whether to include bias

    Returns:
        weights: Dictionary with W and b
    """
    # Xavier/Glorot initialization
    scale = np.sqrt(2.0 / (input_dim + output_dim))

    weights = {
        'W': np.random.randn(input_dim, output_dim) * scale,
    }

    if bias:
        weights['b'] = np.zeros(output_dim)
    else:
        weights['b'] = None

    return weights


def linear(x, weights):
    """
    Apply linear transformation.

    Args:
        x: Input tensor (batch, seq_len, input_dim) or (seq_len, input_dim)
        weights: Dictionary with W and b

    Returns:
        output: Linear transformation result
    """
    W = weights['W']
    b = weights['b']

    # Matrix multiplication: x @ W
    output = np.matmul(x, W)

    # Add bias if present
    if b is not None:
        output = output + b

    return output


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation.

    GELU(x) = x * Phi(x)
    where Phi is the CDF of the standard normal distribution.

    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        output: GELU activation
    """
    # Approximation formula
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    """
    Derivative of GELU activation.

    d/dx GELU(x) = Phi(x) + x * phi(x) + 0.5 * x * phi(x) * t' + 0.5 * tanh(...) * (1 - x * Phi(x))

    This is used for backpropagation.

    Args:
        x: Input to GELU

    Returns:
        derivative: GELU derivative
    """
    # This is approximate - for exact gradients, you'd compute the full derivative
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    c = 0.044715

    t = np.tanh(sqrt_2_over_pi * (x + c * x ** 3))

    # Approximate derivative
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * sqrt_2_over_pi * (1 + 3 * c * x ** 2)


def relu(x):
    """
    ReLU activation: max(0, x)

    Args:
        x: Input tensor

    Returns:
        output: ReLU output
    """
    return np.maximum(0, x)


def tanh(x):
    """
    Tanh activation.

    Args:
        x: Input tensor

    Returns:
        output: Tanh output
    """
    return np.tanh(x)


def feed_forward(x, weights1, weights2, activation=gelu):
    """
    Apply two-layer feed-forward network.

    FFN(x) = activation(x @ W1 + b1) @ W2 + b2

    First layer expands dimension (typically 4x), then contracts back.

    Args:
        x: Input tensor (batch, seq_len, d_model)
        weights1: First linear layer weights
        weights2: Second linear layer weights
        activation: Activation function

    Returns:
        output: Feed-forward output
    """
    # First linear layer
    hidden = linear(x, weights1)

    # Apply activation
    if activation is not None:
        hidden = activation(hidden)

    # Second linear layer
    output = linear(hidden, weights2)

    return output


def feed_forward_with_dropout(x, weights1, weights2, dropout_rate=0.0, training=True):
    """
    Feed-forward with optional dropout.

    Args:
        x: Input
        weights1: First layer weights
        weights2: Second layer weights
        dropout_rate: Dropout probability
        training: Whether in training mode

    Returns:
        output: Output after feed-forward
    """
    output = feed_forward(x, weights1, weights2)

    if training and dropout_rate > 0:
        # Create dropout mask
        mask = np.random.rand(*output.shape) > dropout_rate
        output = output * mask / (1 - dropout_rate)

    return output


class FeedForward:
    """
    Feed-forward network layer.
    """

    def __init__(self, d_model, d_ff=None, activation='gelu'):
        self.d_model = d_model

        if d_ff is None:
            d_ff = d_model * 4

        self.d_ff = d_ff

        self.weights1 = init_linear_weights(d_model, d_ff, bias=True)
        self.weights2 = init_linear_weights(d_ff, d_model, bias=True)

        if activation == 'gelu':
            self.activation = gelu
        elif activation == 'relu':
            self.activation = relu
        elif activation == 'tanh':
            self.activation = tanh
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass."""
        self.last_input = x
        return feed_forward(x, self.weights1, self.weights2, self.activation)

    def backward(self, grad_output):
        """
        Backward pass for feed-forward network.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            grad_input: Gradient w.r.t. input
        """
        batch_size, seq_len, d_model = grad_output.shape
        
        grad_hidden = np.matmul(grad_output, self.weights2['W'].T)
        
        x_ff = np.matmul(self.last_input, self.weights1['W']) + self.weights1['b']
        grad_act = gelu_derivative(x_ff)
        grad_hidden = grad_hidden * grad_act
        
        grad_w1 = np.matmul(self.last_input.transpose(0, 1, 2).reshape(-1, self.d_model).T,
                           np.matmul(grad_output, self.weights2['W'].T).reshape(-1, self.d_ff))
        grad_w1 = grad_w1.reshape(self.d_model, self.d_ff)
        self.weights1['W'] -= 0.001 * grad_w1
        
        grad_w2 = np.matmul(
            (grad_output * gelu_derivative(np.matmul(self.last_input, self.weights1['W']) + self.weights1['b'])).reshape(-1, self.d_ff).T,
            self.last_input.reshape(-1, self.d_model)
        ).T
        self.weights2['W'] -= 0.001 * grad_w2
        
        if self.weights1['b'] is not None:
            grad_b1 = grad_hidden.sum(axis=(0, 1))
            self.weights1['b'] -= 0.001 * grad_b1
        
        if self.weights2['b'] is not None:
            grad_b2 = grad_output.sum(axis=(0, 1))
            self.weights2['b'] -= 0.001 * grad_b2
        
        return np.matmul(grad_hidden, self.weights1['W'].T)

    def get_weights(self):
        """Get weights."""
        return {'weights1': self.weights1, 'weights2': self.weights2}

    def set_weights(self, weights):
        """Set weights."""
        self.weights1 = weights['weights1']
        self.weights2 = weights['weights2']


class Linear:
    """
    Simple linear/dense layer.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = init_linear_weights(input_dim, output_dim, bias)
        self.last_input = None

    def forward(self, x):
        """Forward pass."""
        self.last_input = x
        return linear(x, self.weights)

    def backward(self, grad_output):
        """
        Backward pass for linear layer.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
        
        Returns:
            grad_input: Gradient w.r.t. input
        """
        W = self.weights['W']
        b = self.weights['b']
        
        batch_size = grad_output.shape[0]
        seq_len = grad_output.shape[1] if len(grad_output.shape) > 2 else 1
        
        if len(grad_output.shape) == 3:
            grad_output_2d = grad_output.reshape(batch_size * seq_len, -1)
            x_2d = self.last_input.reshape(batch_size * seq_len, -1)
        else:
            grad_output_2d = grad_output
            x_2d = self.last_input
        
        grad_W = np.matmul(x_2d.T, grad_output_2d)
        
        self.weights['W'] -= 0.001 * grad_W
        
        if b is not None:
            grad_b = np.sum(grad_output_2d, axis=0)
            self.weights['b'] -= 0.001 * grad_b
        
        grad_input = np.matmul(grad_output_2d, W.T)
        
        if len(self.last_input.shape) == 3:
            grad_input = grad_input.reshape(self.last_input.shape)
        
        return grad_input

    def get_weights(self):
        """Get weights."""
        return self.weights

    def set_weights(self, weights):
        """Set weights."""
        self.weights = weights


def linear_backward(x, weights, grad_output):
    """
    Backward pass for linear layer.

    Args:
        x: Input
        weights: Weights
        grad_output: Gradient of loss w.r.t. output

    Returns:
        grad_input: Gradient of loss w.r.t. input
    """
    W = weights['W']
    b = weights['b']

    # Gradient w.r.t. input: grad_input = grad_output @ W^T
    grad_input = np.matmul(grad_output, W.T)

    return grad_input


def gelu_backward(x, grad_output):
    """
    Backward pass for GELU.

    Args:
        x: Input to GELU
        grad_output: Gradient w.r.t. output

    Returns:
        grad_input: Gradient w.r.t. input
    """
    return grad_output * gelu_derivative(x)


def feed_forward_backward(x, weights1, weights2, grad_output):
    """
    Backward pass for feed-forward network.

    Args:
        x: Original input
        weights1: First layer weights
        weights2: Second layer weights
        grad_output: Gradient from next layer

    Returns:
        grad_input: Gradient w.r.t. input
    """
    # This is simplified - proper implementation needs activation derivatives
    grad_hidden = linear_backward(x, weights2, grad_output)
    grad_hidden = grad_hidden * gelu_derivative(np.matmul(x, weights1['W']) + weights1['b'])
    grad_input = linear_backward(x, weights1, grad_hidden)

    return grad_input


class MLP:
    """
    Simple multi-layer perceptron.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass with GELU activation."""
        x = self.linear1.forward(x)
        x = gelu(x)
        x = self.linear2.forward(x)
        return x