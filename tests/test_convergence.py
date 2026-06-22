import numpy as np
from nn.modules import Linear, ReLU
from nn.losses  import CrossEntropyLoss
from nn.optimizers import SGD


def test_linear_network_converges():
    """
    A 2-layer linear network should drive loss to near zero
    on a tiny fixed dataset after enough steps.
    """
    np.random.seed(42)
    X      = np.random.randn(16, 8)
    y      = np.random.randint(0, 4, size=16)

    fc1    = Linear(8, 16)
    relu   = ReLU()
    fc2    = Linear(16, 4)
    loss_fn= CrossEntropyLoss()
    opt    = SGD([fc1, fc2], lr=0.05, momentum=0.9)

    losses = []
    for _ in range(200):
        out       = relu.forward(fc1.forward(X))
        logits    = fc2.forward(out)
        loss, dl  = loss_fn(logits, y)
        losses.append(loss)

        d = fc2.backward(dl)
        d = fc1.backward(relu.backward(d))
        opt.step()
        opt.zero_grad()

    assert losses[-1] < losses[0] * 0.3, (
        f"Loss did not decrease sufficiently: "
        f"{losses[0]:.4f} → {losses[-1]:.4f}"
    )