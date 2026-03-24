import os
from typing import List

import cv2
import numpy as np

from src.common.exceptions import ImageLoadError


def read_image(path: str):
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except Exception as exc:
        raise ImageLoadError(f"Không thể đọc file ảnh: {path}") from exc

    if raw.size == 0:
        raise ImageLoadError(f"File ảnh trống: {path}")

    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"File ảnh không hợp lệ hoặc không hỗ trợ: {path}")
    return img


def write_image(path: str, img) -> None:
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path = f"{path}{ext}"

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ImageLoadError("Không thể encode ảnh để lưu.")

    with open(path, "wb") as file:
        buf.tofile(file)


def list_image_files(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    return files
