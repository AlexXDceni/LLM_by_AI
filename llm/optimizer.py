"""
Optimizer Module
Implements Adam optimizer with gradient clipping and learning rate scheduling.
"""

import numpy as np


def clip_gradients(gradients, max_norm):
    """
    Clip gradients by global norm to prevent exploding gradients.

    Args:
        gradients: Dictionary of gradient arrays
        max_norm: Maximum norm threshold

    Returns:
        clipped_gradients: Dictionary of clipped gradients
    """
    # Compute global norm
    total_norm = 0.0
    for key in gradients:
        if gradients[key] is not None:
            total_norm += np.sum(gradients[key] ** 2)
    total_norm = np.sqrt(total_norm)

    # Clip if needed
    clip_coef = 1.0
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-8)

    clipped = {}
    for key in gradients:
        if gradients[key] is not None:
            clipped[key] = gradients[key] * clip_coef
        else:
            clipped[key] = None

    return clipped


class Adam:
    """
    Adam optimizer with optional weight decay and gradient clipping.
    """

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, clip_norm=None):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm

        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSProp)
        self.t = 0   # Step counter
        selfinitialized = False

    def step(self, parameters, gradients):
        """
        Perform one optimization step.

        Args:
            parameters: Dictionary of model parameters
            gradients: Dictionary of gradients

        Returns:
            updated_parameters: Updated parameters
        """
        self.t += 1

        # Clip gradients if specified
        if self.clip_norm is not None:
            gradients = clip_gradients(gradients, self.clip_norm)

        # Initialize moment vectors on first step
        if not self.initialized:
            for key in parameters:
                if parameters[key] is not None:
                    self.m[key] = np.zeros_like(parameters[key])
                    self.v[key] = np.zeros_like(parameters[key])
            self.initialized = True

        updated = {}
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for key in parameters:
            if parameters[key] is None:
                updated[key] = None
                continue

            p = parameters[key]
            g = gradients.get(key)

            if g is None:
                updated[key] = p
                continue

            # Add weight decay
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            updated[key] = p - lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

        return updated


class AdamW:
    """
    AdamW optimizer with decoupled weight decay.
    """

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, clip_norm=None):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm

        self.m = {}
        self.v = {}
        self.t = 0
        self.initialized = False

    def step(self, parameters, gradients):
        """
        Perform one optimization step with decoupled weight decay.
        """
        self.t += 1

        if self.clip_norm is not None:
            gradients = clip_gradients(gradients, self.clip_norm)

        if not self.initialized:
            for key in parameters:
                if parameters[key] is not None:
                    self.m[key] = np.zeros_like(parameters[key])
                    self.v[key] = np.zeros_like(parameters[key])
            self.initialized = True

        updated = {}
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for key in parameters:
            if parameters[key] is None:
                updated[key] = None
                continue

            p = parameters[key]
            g = gradients.get(key)

            if g is None:
                updated[key] = p
                continue

            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Decoupled weight decay
            updated[key] = p - lr_t * m_hat / (np.sqrt(v_hat) + self.eps) - self.lr * self.weight_decay * p

        return updated


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay.
    """

    def __init__(self, optimizer, warmup_steps=500, total_steps=None,
                 min_lr_ratio=0.1, schedule_type='cosine'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.schedule_type = schedule_type
        self.current_step = 0
        self.base_lr = optimizer.lr

    def step(self):
        """Update learning rate for next step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            if self.schedule_type == 'linear':
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.base_lr * (1 - progress * (1 - self.min_lr_ratio))
            elif self.schedule_type == 'cosine':
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.base_lr * (self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + np.cos(np.pi * progress)))
            else:
                lr = self.base_lr

        self.optimizer.lr = lr
        return lr

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.lr


def get_optimizer(name='adam', **kwargs):
    """
    Factory function to create optimizer by name.

    Args:
        name: Optimizer name ('adam', 'adamw')
        **kwargs: Optimizer parameters

    Returns:
        optimizer: Optimizer instance
    """
    if name.lower() == 'adam':
        return Adam(**kwargs)
    elif name.lower() == 'adamw':
        return AdamW(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")