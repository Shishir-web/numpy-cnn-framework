import numpy as np

class Module:
    def __init__(self):
        self.params = {}
        self.grads  = {}
        self.cache  = {}
        self.training = True

    def forward(self, x): 
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

class Linear(Module):
    """
    Fully connected layer.
    Forward:  out = X @ W + b
    Backward: dX = dout @ W.T
               dW = X.T @ dout
               db = dout.sum(axis=0)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # He initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.params["W"] = np.random.randn(in_features, out_features) * scale
        self.params["b"] = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["X"] = x
        return x @ self.params["W"] + self.params["b"]
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["X"]
        self.grads["W"] = x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T
    
class ReLU(Module):
    """
     ReLU activation.
    Forward:  out = max(0, x)
    Backward: dX  = dout * (x > 0)
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.eps      = eps
        self.momentum = momentum
        self.params["gamma"] = np.ones(num_features)
        self.params["beta"]  = np.zeros(num_features)
        self.running_mean    = np.zeros(num_features)
        self.running_var     = np.ones(num_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape

        if self.training:
            # Mean and variance over N, H, W
            mean = x.mean(axis=(0, 2, 3))              #(C,)
            var =  x.var( axis=(0, 2, 3))              #(C,)

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean)
            self.running_var   = ((1 - self.momentum) * self.running_var + self.momentum * var)
        else:
            mean = self.running_mean
            var  = self.running_var
        
        x_norm = (x - mean[None, :, None, None]) / np.sqrt(var[:, None, None, None] + self.eps)

        out    = (self.params["gamma"][None, :, None, None] * x_norm
                  + self.params["beta"][None, :, None, None])
        
        if self.training:
            self.cache.update({
                "x": x,"x_norm": x_norm,
                "mean": mean,
                "var": var, "N": N, "H": H, "W": W,
            })
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dL/dgamma = sum_i (dL/dout)
        x      = self.cache["x"]
        x_norm = self.cache["x_norm"]
        mean   = self.cache["mean"]
        var   = self.cache["var"]
        N, H, W = self.cache["N"], self.cache["H"], self.cache["W"]
        M =  N * H * W

        gamma = self.params["gamma"]

        self.grads["gamma"] = (dout * x_norm).sum(axis=(0, 2, 3))
        self.grads["beta"]  = dout.sum(axis=(0, 2, 3))

        dx_norm = dout * gamma[None, :, None, None]

        dvar  = (dx_norm * (x - mean[None, :, None, None]) *
                 -0.5 * (var[None, : None, None] + self.eps) ** -1.5).sum(axis=(0, 2, 3))
        
        dmean = ((-dx_norm / np.sqrt(var[None, : None, None] + self.eps)).sum(axis=(0, 2, 3)) *
                 + dvar * (-2 *(x - mean[None, :, None, None])
                 ).mean(axis=(0, 2, 3)))
        
        dx =    (dx_norm / np.sqrt(var[None, :, None, None] + self.eps)
                 + dvar[None, :, None, None]
                 * 2 * (x - mean[None, :, None, None]) / M
                 + dmean[None, :, None, None] / M)
         
        return dx
    
class Softmax(Module):
    """Numerically stable softmax (used at inference only —
    during training CrossEntropyLoss fuses softmax + log).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use CrossEntropyLoss for training.")
        