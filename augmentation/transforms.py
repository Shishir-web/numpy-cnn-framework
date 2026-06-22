import numpy as np


def _apply_affine(image: np.ndarray,
                  matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 2x3 affine transformation matrix to a (C, H, W) image
    using bilinear interpolation — fully vectorised.
    """
    C, H, W = image.shape
    out     = np.zeros_like(image)

    # Build output pixel grid
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ones    = np.ones_like(xs)
    coords  = np.stack([xs, ys, ones], axis=-1)       # (H, W, 3)

    # Map output coords to input coords
    src = coords @ matrix.T                            # (H, W, 2)
    src_x, src_y = src[..., 0], src[..., 1]

    # Bilinear interpolation
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    # Clip to valid range
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Weights
    wx = src_x - x0
    wy = src_y - y0

    for c in range(C):
        Ia = image[c][y0c, x0c]
        Ib = image[c][y1c, x0c]
        Ic = image[c][y0c, x1c]
        Id = image[c][y1c, x1c]
        out[c] = (Ia * (1 - wx) * (1 - wy)
                  + Ib * (1 - wx) * wy
                  + Ic * wx * (1 - wy)
                  + Id * wx * wy)
    return out


def random_rotation(image: np.ndarray,
                    max_angle: float = 15.0) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    theta = np.radians(angle)
    C, H, W = image.shape
    cx, cy  = W / 2, H / 2

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Affine matrix: rotate around centre
    M = np.array([
        [cos_t,  sin_t, cx - cx * cos_t - cy * sin_t],
        [-sin_t, cos_t, cy + cx * sin_t - cy * cos_t],
    ])
    return _apply_affine(image, M)


def random_translation(image: np.ndarray,
                       max_shift: float = 4.0) -> np.ndarray:
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)
    M  = np.array([
        [1, 0, tx],
        [0, 1, ty],
    ])
    return _apply_affine(image, M)


def random_shear(image: np.ndarray,
                 max_shear: float = 0.1) -> np.ndarray:
    shear = np.random.uniform(-max_shear, max_shear)
    M     = np.array([
        [1,     shear, 0],
        [0,     1,     0],
    ])
    return _apply_affine(image, M)


def random_horizontal_flip(image: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        return image[:, :, ::-1].copy()
    return image


def color_jitter(image: np.ndarray,
                 brightness: float = 0.2,
                 contrast:   float = 0.2,
                 saturation: float = 0.1) -> np.ndarray:
    """Apply random brightness, contrast, saturation jitter."""
    out = image.copy()

    # Brightness
    b   = np.random.uniform(1 - brightness, 1 + brightness)
    out = out * b

    # Contrast — scale around channel mean
    c    = np.random.uniform(1 - contrast, 1 + contrast)
    mean = out.mean(axis=(1, 2), keepdims=True)
    out  = (out - mean) * c + mean

    # Saturation — blend with grayscale version
    s      = np.random.uniform(1 - saturation, 1 + saturation)
    gray   = out.mean(axis=0, keepdims=True)
    out    = s * out + (1 - s) * gray

    return np.clip(out, 0, 1)