"""
PyTorch baseline — equivalent architecture to our NumPy CNN.
Run both and compare test accuracy to verify gradient correctness.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_baseline(epochs=10):
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
        transforms.RandomHorizontalFlip(),
    ])

    train_ds = datasets.CIFAR10(
        "data/", train=True,  download=True, transform=transform
    )
    test_ds  = datasets.CIFAR10(
        "data/", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256)

    model     = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                test_correct += (out.argmax(1) == y).sum().item()
                test_total   += y.size(0)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"train_acc={correct/total:.3f} | "
              f"test_acc={test_correct/test_total:.3f}")


if __name__ == "__main__":
    train_baseline(epochs=10)