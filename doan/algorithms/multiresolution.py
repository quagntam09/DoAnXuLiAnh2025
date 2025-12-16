import os
import cv2
import numpy as np
from typing import Callable, List, Tuple

import time

from .average_color import extract
from .kdtree_nn import KDTreeNearestNeighbor


def level_sizes(base_tile: int, levels: int = 3) -> List[int]:
    if base_tile <= 0:
        raise ValueError("base_tile phải > 0")
    if levels <= 0:
        raise ValueError("levels phải > 0")

    sizes = []
    s = base_tile * (2 ** (levels - 1))
    for _ in range(levels):
        sizes.append(int(s))
        s //= 2
        if s <= 0:
            break
    return sizes


def _list_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []

    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))

    if not files:
        raise Exception(f"Không tìm thấy ảnh trong thư mục: {folder}")

    return files


def prepare_tiles_for_level(tiles_folder: str, tile_size: int,
                            progress_callback: Callable[[float, str], None] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:

    if progress_callback is None:
        progress_callback = lambda p, msg: None

    file_list = _list_image_files(tiles_folder)
    if not file_list:
        raise Exception("Folder tiles trống hoặc không có ảnh hợp lệ!")

    tiles = []
    colors = []
    total = len(file_list)

    progress_callback(0, f"Đang chuẩn bị tiles cho level tile={tile_size}...")

    for i, path in enumerate(file_list):
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            img_small = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        except Exception:
            continue

        tiles.append(img_small)
        colors.append(extract(img_small))  # (3,)

        if i % 100 == 0:
            progress_callback((i / max(1, total)) * 100, f"Tiles level {tile_size}: {i}/{total}")

    if not tiles:
        raise Exception("Không tìm thấy tile hợp lệ sau khi đọc!")

    tiles_arr = np.array(tiles, dtype=np.uint8)
    colors_arr = np.array(colors, dtype=np.float32)
    return tiles_arr, colors_arr


def build_index(colors_arr: np.ndarray):
    return KDTreeNearestNeighbor(colors_arr)


def render_single_level(target_img, tiles_arr, index, tile_size,
                        progress_callback=None, frame_callback=None,
                        frame_every=200, max_fps=15):
    if progress_callback is None:
        progress_callback = lambda p, msg: None

    h, w = target_img.shape[:2]
    mosaic = target_img.copy()

    rows = range(0, h, tile_size)
    cols = range(0, w, tile_size)
    total_blocks = (h // tile_size) * (w // tile_size)
    done = 0

    min_dt = 1.0 / max(1, max_fps)
    last_push = 0.0

    for y in rows:
        for x in cols:
            roi = target_img[y:y + tile_size, x:x + tile_size]
            avg = extract(roi)
            idx = index.query(avg)
            mosaic[y:y + tile_size, x:x + tile_size] = tiles_arr[idx]

            done += 1

            if done % 300 == 0:
                progress_callback((done / max(1, total_blocks)) * 100,
                                    f"Đang ghép level {tile_size}: {done}/{total_blocks}")

            if frame_callback is not None and (done % max(1, frame_every) == 0):
                now = time.time()
                if (now - last_push) >= min_dt:
                    last_push = now
                    frame_callback(mosaic)

    if frame_callback is not None:
        frame_callback(mosaic)

    return mosaic


def multi_resolution_mosaic(
    target_path: str,
    tiles_folder: str,
    base_tile: int = 10,
    levels: int = 3,
    blend_factor: float = 0.0,
    progress_callback: Callable[[float, str], None] = None,
    frame_callback=None,
    frame_every: int = 200
) -> Tuple[np.ndarray, List[int]]:

    if progress_callback is None:
        progress_callback = lambda p, msg: None

    # --- 1. Đọc ảnh gốc ---
    target = cv2.imread(target_path)
    if target is None:
        raise Exception("Lỗi đọc ảnh gốc")

    # --- 2. Tính các level (thô → mịn) ---
    sizes = level_sizes(base_tile, levels)
    progress_callback(1, f"Multi-resolution sizes: {sizes}")

    final_mosaic = None

    # --- 3. Chạy từng level ---
    for li, ts in enumerate(sizes):
        h, w = target.shape[:2]
        h_new = (h // ts) * ts
        w_new = (w // ts) * ts
        target_crop = target[:h_new, :w_new]

        # --- 3.1 Chuẩn bị tiles (LUÔN LUÔN build mới) ---
        progress_callback(5 + li * 10, f"Chuẩn bị tiles level {ts}...")
        tiles_arr, colors_arr = prepare_tiles_for_level(tiles_folder, ts)

        # --- 3.2 Build KD-tree ---
        progress_callback(20 + li * 10, f"Build KD-tree level {ts}...")
        index = build_index(colors_arr)

        # --- 3.3 Render mosaic ---
        progress_callback(30 + li * 10, f"Render mosaic level {ts}...")
        mosaic_lv = render_single_level(
            target_crop,
            tiles_arr,
            index,
            ts,
            frame_callback=frame_callback,
            frame_every=frame_every,
            max_fps=15
        )

        final_mosaic = mosaic_lv
        progress_callback(40 + li * 15, f"Done level {ts}")

    # --- 4. Blend với ảnh gốc (nếu có) ---
    if blend_factor > 0:
        progress_callback(95, "Blending...")
        ts_last = sizes[-1]
        h, w = target.shape[:2]
        h_new = (h // ts_last) * ts_last
        w_new = (w // ts_last) * ts_last
        target_crop = target[:h_new, :w_new]

        final_mosaic = cv2.addWeighted(
            final_mosaic,
            1.0 - blend_factor,
            target_crop,
            blend_factor,
            0
        )

    progress_callback(100, "Hoàn tất multi-resolution!")
    return final_mosaic, sizes
