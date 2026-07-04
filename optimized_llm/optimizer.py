"""
optimizer.py — AdamW optimizer with gradient clipping.
C++: struct AdamWState { float *m, *v; float lr, beta1, beta2, eps, wd; };
"""

import numpy as np


def clip_gradients(gradients: dict[str, np.ndarray],
                   max_norm: float) -> dict[str, np.ndarray]:
    """Clip gradients by global L2 norm."""
    total_norm = 0.0
    for g in gradients.values():
        if g is not None:
            total_norm += float(np.sum(g ** 2))
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        return {k: (v * scale if v is not None else None)
                for k, v in gradients.items()}
    return gradients


class AdamW:
    """
    AdamW with decoupled weight decay.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01,
                 max_norm: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_norm = max_norm

        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t: int = 0
        self.initialized: bool = False

    def step(self, params: dict[str, np.ndarray],
             grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Perform one optimization step.
        Returns updated parameters.
        """
        self.t += 1

        grads = clip_gradients(grads, self.max_norm)

        if not self.initialized:
            for k, p in params.items():
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
            self.initialized = True

        bias_corr1 = 1.0 - self.beta1 ** self.t
        bias_corr2 = 1.0 - self.beta2 ** self.t
        lr_t = self.lr * np.sqrt(bias_corr2) / bias_corr1

        updated = {}
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                updated[k] = p
                continue

            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[k] / bias_corr1
            v_hat = self.v[k] / bias_corr2

            # Decoupled weight decay
            updated[k] = (p - lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
                          - self.lr * self.weight_decay * p)

        return updated
