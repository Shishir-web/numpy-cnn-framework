import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.cifar10 import load_cifar10, get_batches
from nn           import Sequential, ReLU, Linear, BatchNorm2D, SGD
from nn.conv      import Conv2D
from nn.pooling   import MaxPool2D
from nn.losses    import CrossEntropyLoss
from augmentation import AugmentationPipeline
from tqdm import tqdm


def build_model():
    """
    Small CNN for CIFAR-10:
    Conv(32) → BN → ReLU → Pool →
    Conv(64) → BN → ReLU → Pool →
    Flatten → Linear(256) → ReLU → Linear(10)
    """
    return Sequential(
        Conv2D(3,  32, kernel_size=3, padding=1),
        BatchNorm2D(32),
        ReLU(),
        MaxPool2D(2, 2),

        Conv2D(32, 64, kernel_size=3, padding=1),
        BatchNorm2D(64),
        ReLU(),
        MaxPool2D(2, 2),
    )


class FlattenLayer:
    params = {}
    grads  = {}

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)

    def __call__(self, x):
        return self.forward(x)

    def train(self): pass
    def eval(self):  pass


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train(epochs=10, batch_size=64, lr=0.01):
    print("Loading CIFAR-10...")
    X_train, y_train, X_test, y_test = load_cifar10()
    augment = AugmentationPipeline(p=0.5)
    loss_fn = CrossEntropyLoss()

    # Build full model manually
    conv1  = Conv2D(3,  32, kernel_size=3, padding=1)
    bn1    = BatchNorm2D(32)
    relu1  = ReLU()
    pool1  = MaxPool2D(2, 2)
    conv2  = Conv2D(32, 64, kernel_size=3, padding=1)
    bn2    = BatchNorm2D(64)
    relu2  = ReLU()
    pool2  = MaxPool2D(2, 2)
    flat   = FlattenLayer()
    fc1    = Linear(64 * 8 * 8, 256)
    relu3  = ReLU()
    fc2    = Linear(256, 10)

    layers = [conv1, bn1, relu1, pool1,
              conv2, bn2, relu2, pool2,
              flat, fc1, relu3, fc2]

    modules_with_params = [conv1, bn1, conv2, bn2, fc1, fc2]
    optimizer = SGD(modules_with_params, lr=lr, momentum=0.9)

    for epoch in range(epochs):
        total_loss, total_acc, n_batches = 0, 0, 0

        for X_batch, y_batch in get_batches(
            X_train, y_train, batch_size=batch_size
        ):
            X_batch = augment(X_batch)

            # Forward
            x = X_batch
            for layer in layers:
                x = layer.forward(x)
            logits = x

            loss, dlogits = loss_fn(logits, y_batch)

            # Backward
            d = dlogits
            for layer in reversed(layers):
                d = layer.backward(d)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss
            total_acc  += accuracy(logits, y_batch)
            n_batches  += 1

        # Eval
        for layer in layers:
            if hasattr(layer, "eval"):
                layer.eval()

        test_acc_total, test_batches = 0, 0
        for X_batch, y_batch in get_batches(
            X_test, y_test, batch_size=256, shuffle=False
        ):
            x = X_batch
            for layer in layers:
                x = layer.forward(x)
            test_acc_total += accuracy(x, y_batch)
            test_batches   += 1

        for layer in layers:
            if hasattr(layer, "train"):
                layer.train()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"loss={total_loss/n_batches:.4f} | "
              f"train_acc={total_acc/n_batches:.3f} | "
              f"test_acc={test_acc_total/test_batches:.3f}")


if __name__ == "__main__":
    train(epochs=10)