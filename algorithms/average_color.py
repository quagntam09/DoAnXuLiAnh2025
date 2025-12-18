import numpy as np
import cv2

def extract(img: np.ndarray, use_texture=False) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")
    
    # 1. Tính màu trung bình (Mean)
    mean_color = np.mean(img, axis=(0, 1)) # (3,)

    if not use_texture:
        return mean_color.astype(np.float32)

    # 2. Tính độ lệch chuẩn (Std Dev) để đại diện cho texture
    # Ảnh nhiều chi tiết -> std cao, ảnh trơn -> std thấp
    std_color = np.std(img, axis=(0, 1)) # (3,)

    # Kết hợp thành vector 6 chiều (cân trọng số nếu cần)
    feature = np.concatenate([mean_color, std_color]) 
    return feature.astype(np.float32)