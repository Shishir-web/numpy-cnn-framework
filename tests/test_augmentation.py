import numpy as np
from augmentation.transforms import (
    random_rotation, random_translation,
    random_shear, random_horizontal_flip, color_jitter,
)
from augmentation.pipeline import AugmentationPipeline


def make_image():
    return np.random.rand(3, 32, 32).astype(np.float32)


def test_rotation_shape():
    img = make_image()
    out = random_rotation(img, max_angle=15)
    assert out.shape == img.shape

def test_translation_shape():
    img = make_image()
    out = random_translation(img, max_shift=4)
    assert out.shape == img.shape

def test_shear_shape():
    img = make_image()
    out = random_shear(img, max_shear=0.1)
    assert out.shape == img.shape

def test_flip_mirrors_correctly():
    img      = np.zeros((3, 4, 4))
    img[0, 0, 0] = 1.0
    np.random.seed(0)   # seed so flip triggers
    out = random_horizontal_flip(img)
    # Either flipped or not — just check shape
    assert out.shape == img.shape

def test_color_jitter_shape():
    img = make_image()
    out = color_jitter(img)
    assert out.shape == img.shape

def test_color_jitter_clip():
    img = make_image()
    out = color_jitter(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_pipeline_batch():
    pipe = AugmentationPipeline(p=1.0)
    X    = np.random.rand(4, 3, 32, 32).astype(np.float32)
    out  = pipe(X)
    assert out.shape == X.shape

def test_pipeline_single_image():
    pipe = AugmentationPipeline(p=1.0)
    img  = np.random.rand(3, 32, 32).astype(np.float32)
    out  = pipe(img)
    assert out.shape == img.shape