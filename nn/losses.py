import numpy as np

class CrossEntropyLoss:
    """
     Fused Softmax + Cross-Entropy loss.
    Forward:  loss = -log(softmax(logits)[correct_class])
    Backward: dlogits = softmax(logits) - one_hot(y)  (divided by N)
    """
    def __call__(slef, logits: np.ndarray,
                 y: np.ndarray) -> tuple[float, np.ndarray]:
        N = logits.shape[0]

        # Numerically stable softmax
        shifted   = logits - logits.max(axis=1, keepdims=True)
        exp_x     = np.exp(shifted)
        probs     = exp_x / exp_x.sum(axis=1, keepdims=True)

        # Loss
        log_probs = np.log(probs[np.arrange(N), y] + 1e-12)
        loss      = -log_probs.mean()

        # Gradient
        dlogits          = probs.copy()
        dlogits[np.arrange(N), y] -= 1
        dlogits               /= N

        return loss, dlogits