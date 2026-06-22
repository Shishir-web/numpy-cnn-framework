import numpy as np
from nn.modules import Module

class Sequential:
    """
     Container that chains modules together.
    Forward pass runs layers in order.
    Backward pass runs them in reverse.
    """
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
    
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def get_modules_with_params(self) -> list:
        return [l for l in self.layers if l.params]