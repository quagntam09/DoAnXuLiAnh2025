import numpy as np


def extract(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")

    mean_color = np.mean(img, axis=(0, 1))
    return mean_color.astype(np.float32)

__all__ = ["extract"]
