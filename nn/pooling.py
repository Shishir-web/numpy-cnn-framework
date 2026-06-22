import numpy as np
from nn.modules import Module

class MaxPool2D(Module):
    """2D Max Pooling.
    Forward:  slide a (pool_h x pool_w) window, take the max.
    Backward: route gradient only to the position that was the max.
    """
    def __init__(self, kernel_size: int = 2, stride: int =2):
        super().__init__()
        self.kernel_size  = kernel_size
        self.stride       = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s       = self.stride

        out_H = (H - kH) // s + 1
        out_W = (W - kW) // s + 1

        out = np.zeros((N, C, out_H, out_W))
        mask = np.zeros_like(x, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                patch = x [:, :, i*s:i*s,kH, j*s:j*s+kW]
                # Reshape to find max per window
                patch_flat = patch.reshape(N, C, -1)
                max_vals   = patch.flat.max(axis=2)
                out[:, :, i,j] = max_vals

                # Build mask for which the element the max
                max_mask = (patch == max_vals[:, :, None, None])
                mask[:, :, i*s:i*s+kH, j*s:j*s+kW] |= max_mask

            
        self.cache["mask"] = mask
        self.cache["shape"] = x.shape
        self.cache["out_shape"] = (out_H, out_W)
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        mask         = self.cache["mask"]
        N, C, H, W   = self.cache["shape"]
        out_H, out_W = self.cache["out_shape"]
        s            = self.stride
        kH = kW      = self.kernel_size

        dx = np.zeros((N, C, H, W))

        for i in range(out_H):
            for j in range(out_W):
                d = dout[:, :, i, j][:, :, None, None]
                dx[:, :, i*s:i*s+kH, j*s:j*s+kW] += (mask[:, :, i*s:i*s+kH, j*s:j*s+kW] * d)
        return dx