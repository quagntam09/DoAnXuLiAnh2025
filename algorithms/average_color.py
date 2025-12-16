import numpy as np

def extract(img: np.ndarray) -> np.ndarray:
    """
    Tính màu trung bình (B, G, R) của một ảnh.
    Input: img (numpy array HxWxC)
    Output: numpy array (3,)
    """
    if img is None:
        raise ValueError("Image is None")

    if img.ndim != 3 or img.shape[2] != 3:
        # Nếu là ảnh xám, convert dimension để xử lý thống nhất
        if img.ndim == 2:
            return np.mean(img).repeat(3).astype(np.float32)
        raise ValueError(f"Invalid image shape: {img.shape}")

    # Tính mean theo trục H và W (axis 0 và 1)
    avg_color = np.mean(img, axis=(0, 1))

    return avg_color.astype(np.float32)