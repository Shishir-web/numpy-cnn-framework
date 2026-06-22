import numpy as np
from nn.modules import Module


class Conv2D(Module):
    """
    2D Convolutional layer.

    Forward:  out[n,f,i,j] = sum over c,kh,kw of
                              W[f,c,kh,kw] * x[n,c,i+kh,j+kw] + b[f]

    Backward: dW = conv(x, dout)
              db = dout.sum over n,i,j
              dx = full_conv(dout, flip(W))

    Uses im2col for efficiency.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        # He initialisation
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.params["W"] = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.params["b"] = np.zeros(out_channels)

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input patches to columns for efficient convolution.
        x: (N, C, H, W)
        returns: (N * out_H * out_W, C * kH * kW)
        """
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s, p     = self.stride, self.padding

        out_H = (H + 2 * p - kH) // s + 1
        out_W = (W + 2 * p - kW) // s + 1

        x_pad = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode="constant")

        col = np.zeros((N, C, kH, kW, out_H, out_W))
        for i in range(kH):
            i_max = i + s * out_H
            for j in range(kW):
                j_max = j + s * out_W
                col[:, :, i, j, :, :] = x_pad[
                    :, :, i:i_max:s, j:j_max:s
                ]

        col = col.transpose(0, 4, 5, 1, 2, 3)
        col = col.reshape(N * out_H * out_W, -1)
        return col, out_H, out_W

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        kH = kW    = self.kernel_size
        F          = self.out_channels

        col, out_H, out_W = self._im2col(x)

        W_col = self.params["W"].reshape(F, -1)
        out   = col @ W_col.T + self.params["b"]
        out   = out.reshape(N, out_H, out_W, F).transpose(0, 3, 1, 2)

        self.cache["x"]     = x
        self.cache["col"]   = col
        self.cache["out_H"] = out_H
        self.cache["out_W"] = out_W
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x     = self.cache["x"]
        col   = self.cache["col"]
        out_H = self.cache["out_H"]
        out_W = self.cache["out_W"]
        N, C, H, W = x.shape
        F          = self.out_channels
        kH = kW    = self.kernel_size
        s, p       = self.stride, self.padding

        # dout: (N, F, out_H, out_W) → (N*out_H*out_W, F)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, F)

        self.grads["b"] = dout_flat.sum(axis=0)
        self.grads["W"] = (dout_flat.T @ col).reshape(self.params["W"].shape)

        W_col  = self.params["W"].reshape(F, -1)
        dcol   = dout_flat @ W_col            # (N*out_H*out_W, C*kH*kW)

        # col2im — reverse of im2col
        dcol   = dcol.reshape(N, out_H, out_W, C, kH, kW)
        dcol   = dcol.transpose(0, 3, 4, 5, 1, 2)

        H_pad  = H + 2 * p
        W_pad  = W + 2 * p
        dx_pad = np.zeros((N, C, H_pad, W_pad))

        for i in range(kH):
            i_max = i + s * out_H
            for j in range(kW):
                j_max = j + s * out_W
                dx_pad[:, :, i:i_max:s, j:j_max:s] += dcol[:, :, i, j, :, :]

        dx = dx_pad[:, :, p:p+H, p:p+W] if p > 0 else dx_pad
        return dx