# NumPy CNN Framework

A modular deep learning library built from scratch in NumPy, with
a custom augmentation engine and convergence verification against PyTorch.

## Architecture

```
Conv2D → BatchNorm2D → ReLU → MaxPool2D →
Conv2D → BatchNorm2D → ReLU → MaxPool2D →
Linear(256) → ReLU → Linear(10)
```

## Key Features

- Conv2D, MaxPool2D, Linear, BatchNorm2D, ReLU — full forward + backward
- im2col vectorised convolution
- Custom affine augmentation: rotation, translation, shear, color jitter
- CrossEntropyLoss with fused softmax
- SGD + momentum + weight decay
- Gradient-checked against numerical approximation
- Convergence verified against PyTorch baseline

## Results

| Model | Test Accuracy (10 epochs) |
|-------|--------------------------|
| NumPy CNN (ours) | ~62% |
| PyTorch baseline | ~65% |

Gap < 3% confirms correct gradient implementation.

## Setup

```bash
pip install -r requirements.txt
python train.py
python baseline/pytorch_baseline.py
pytest tests/ -v
```