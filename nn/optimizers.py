import numpy as np
from typing import list as List

class SGD:
    """Stochastic Gradient Descent with momentum.
    v = momentum * v - lr * grad
    param += v
    """
    def __init__(self, modules: list, lr: float = 0.01,
        momentum: float = 0.9, weight_decay: float = 1e-4):
        self.modules = modules
        self.lr       = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # initialise velocity buffers
        self.velocities = []
        for m in modules:
            v = {}
            for k in m.params:
                v[k] = np.zeros_like(m.params[k])
            self.velocities.append(v)

    def step(self):
        for m,  v in zip(self.modules, self.velocities):
            for k in m.params:
                grad = m.grads.get(k, np.zeros_like(m.params[k]))
                # weight decay
                grad = grad + self.weight_decay * m.params[k]
                v[k] = self.momentum * v[k] - self.lr * grad
                m.params[k] += v[k]

    def zero_grad(self):
        for m in self.modules:
            for k in m.grads:
                m.grads[k] = np.zeros_like(m.params[k])
