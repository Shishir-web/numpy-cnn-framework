import numpy as np
from augmentation.transforms import (
    random_rotation, random_translation,
    random_shear, random_horizontal_flip, color_jitter,
)

class AugmentationPipeline:
    """Composable augmentation pipeline.
    Each transform is applied with a given probability.
    """
    def __init__(self,
                 rotation:    bool = True,
                 translation: bool = True,
                 shear:       bool = True,
                 flip:        bool = True,
                 jitter:      bool = True,
                 p:           float= 0.5):
        self.rotation     = rotation
        self.translation  = translation
        self.shear        = shear
        self.flip         = flip
        self.jitter       = jitter
        self.p            = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ x: (N, C, H, W) batch or (C, H, W) single image.
        """
        single = x.ndim == 3
        if single:
            x = x[None]

        out = []
        for img in x:
            if self.flip          and np.random.rand() < self.p:
                img = random_horizontal_flip(img)
            if self.rotation      and np.random.rand() < self.p:
                img = random_rotation(img)
            if self.translation   and np.random.rand() < self.p:
                img = random_translation(img)
            if self.shear         and np.random.rand() < self.p:
                img = random_shear(img)
            if self.jitter        and np.random.rand() < self.p:
                img = color_jitter(img)
            out.append(img)

        result = np.stack(out)
        return result[0] if single else result