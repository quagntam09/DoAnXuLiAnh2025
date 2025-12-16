import os
import cv2
import numpy as np
from typing import List, Tuple

def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str,
                        exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                        ) -> List[str]:
    """Liệt kê danh sách file ảnh hợp lệ trong folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder không tồn tại: {folder}")

    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(exts):
            files.append(os.path.join(folder, name))
    return files

def read_image(path: str) -> np.ndarray:
    """Đọc ảnh bằng OpenCV, ném lỗi nếu đọc thất bại."""
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Không đọc được ảnh: {path}")
    return img


def safe_read_image(path: str) -> np.ndarray | None:
    """Đọc ảnh an toàn, trả None nếu lỗi."""
    try:
        return cv2.imread(path)
    except Exception:
        return None


def write_image(path: str, img: np.ndarray):
    """Ghi ảnh ra file, tự tạo folder nếu cần."""
    ensure_dir(os.path.dirname(path) or ".")
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Ghi ảnh thất bại: {path}")

def crop_to_multiple(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Crop ảnh sao cho H, W chia hết cho tile_size."""
    if tile_size <= 0:
        raise ValueError("tile_size phải > 0")

    h, w = img.shape[:2]
    h_new = (h // tile_size) * tile_size
    w_new = (w // tile_size) * tile_size
    return img[:h_new, :w_new]


def resize_tile(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Resize tile về kích thước (tile_size, tile_size)."""
    return cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

def load_and_prepare_tiles(tiles_folder: str, tile_size: int):
    """
    Đọc toàn bộ ảnh tile trong folder, resize về tile_size.
    Trả về:
        tiles_arr: (N, tile_size, tile_size, 3) uint8
    """
    files = list_image_files(tiles_folder)
    if not files:
        raise Exception("Không tìm thấy ảnh tile hợp lệ!")

    tiles = []
    for path in files:
        img = safe_read_image(path)
        if img is None:
            continue
        try:
            tiles.append(resize_tile(img, tile_size))
        except Exception:
            continue

    if not tiles:
        raise Exception("Tất cả ảnh tile đều lỗi!")

    return np.array(tiles, dtype=np.uint8)
