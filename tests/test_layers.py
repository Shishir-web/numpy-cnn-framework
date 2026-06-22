import numpy as np
import pytest
from nn.modules import Linear, ReLU, BatchNorm2D
from nn.conv    import Conv2D


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it   = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx       = it.multi_index
        orig      = x[idx]
        x[idx]    = orig + eps
        fp        = f(x)
        x[idx]    = orig - eps
        fm        = f(x)
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx]    = orig
        it.iternext()
    return grad


# ── Linear ───────────────────────────────────────────────────

def test_linear_forward_shape():
    layer = Linear(4, 8)
    x     = np.random.randn(3, 4)
    out   = layer.forward(x)
    assert out.shape == (3, 8)

def test_linear_backward_shape():
    layer = Linear(4, 8)
    x     = np.random.randn(3, 4)
    layer.forward(x)
    dx = layer.backward(np.ones((3, 8)))
    assert dx.shape == (3, 4)

def test_linear_gradient_check():
    np.random.seed(0)
    layer = Linear(4, 6)
    x     = np.random.randn(2, 4)

    def f(x_):
        return layer.forward(x_.copy()).sum()

    layer.forward(x)
    dx_analytic = layer.backward(np.ones((2, 6)))
    dx_numeric  = numerical_gradient(f, x.copy())
    np.testing.assert_allclose(dx_analytic, dx_numeric, atol=1e-4)


# ── ReLU ─────────────────────────────────────────────────────

def test_relu_forward():
    relu = ReLU()
    x    = np.array([-1.0, 0.0, 1.0, 2.0])
    out  = relu.forward(x)
    np.testing.assert_array_equal(out, [0, 0, 1, 2])

def test_relu_backward():
    relu = ReLU()
    x    = np.array([-1.0, 0.5, 2.0])
    relu.forward(x)
    dx = relu.backward(np.ones(3))
    np.testing.assert_array_equal(dx, [0, 1, 1])


# ── Conv2D ───────────────────────────────────────────────────

def test_conv2d_forward_shape():
    conv = Conv2D(3, 8, kernel_size=3, padding=1)
    x    = np.random.randn(2, 3, 8, 8)
    out  = conv.forward(x)
    assert out.shape == (2, 8, 8, 8)

def test_conv2d_gradient_check():
    np.random.seed(1)
    conv = Conv2D(2, 3, kernel_size=3, padding=1)
    x    = np.random.randn(2, 2, 6, 6) * 0.1

    def f(x_):
        return conv.forward(x_.copy()).sum()

    conv.forward(x)
    dout        = np.ones_like(conv.forward(x))
    dx_analytic = conv.backward(dout)
    dx_numeric  = numerical_gradient(f, x.copy())
    np.testing.assert_allclose(dx_analytic, dx_numeric, atol=1e-3)