import numpy as np
from nn.pooling import MaxPool2D


def test_maxpool_forward_shape():
    pool = MaxPool2D(2, 2)
    x    = np.random.randn(2, 4, 8, 8)
    out  = pool.forward(x)
    assert out.shape == (2, 4, 4, 4)

def test_maxpool_forward_values():
    pool = MaxPool2D(2, 2)
    x    = np.array([[[[1, 3], [2, 4]]]], dtype=float)
    out  = pool.forward(x)
    assert out[0, 0, 0, 0] == 4.0

def test_maxpool_backward_shape():
    pool = MaxPool2D(2, 2)
    x    = np.random.randn(2, 4, 8, 8)
    pool.forward(x)
    dx = pool.backward(np.ones((2, 4, 4, 4)))
    assert dx.shape == x.shape

def test_maxpool_gradient_routes_to_max():
    pool  = MaxPool2D(2, 2)
    x     = np.array([[[[1.0, 3.0], [2.0, 4.0]]]])
    pool.forward(x)
    dx    = pool.backward(np.ones((1, 1, 1, 1)))
    assert dx[0, 0, 1, 1] == 1.0   # max was at position (1,1)
    assert dx[0, 0, 0, 0] == 0.0