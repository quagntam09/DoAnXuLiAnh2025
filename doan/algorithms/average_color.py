import numpy as np

def extract(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

    avg_color = np.mean(img, axis=(0, 1))

    return avg_color.astype(np.float32)
