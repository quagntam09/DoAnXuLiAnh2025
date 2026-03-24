import concurrent.futures
import hashlib
import os
import time
from collections import deque
from typing import Callable, List, Tuple

import cv2
import numpy as np

from src.common.config import DEFAULT_SPLIT_THRESHOLD
from src.common.exceptions import ImageLoadError, TileDatasetError
from src.core.algorithms.kdtree_nn import KDTreeNearestNeighbor
from src.infra.image_io import list_image_files, read_image

CACHE_VERSION = "v2"
CACHE_DIR = os.path.join(".cache", "mosaic_tiles")


def level_sizes(base_tile: int, levels: int = 3) -> List[int]:
    if base_tile <= 0:
        raise ValueError("base_tile phải > 0")
    if levels <= 0:
        raise ValueError("levels phải > 0")

    sizes = []
    max_s = base_tile * (2 ** (levels - 1))
    current = max_s
    while current >= base_tile:
        sizes.append(current)
        current //= 2
    return sizes


def _compute_features_batch(tiles: np.ndarray) -> np.ndarray:
    return tiles.mean(axis=(1, 2), dtype=np.float32)


def _dataset_fingerprint(file_list: List[str]) -> str:
    hasher = hashlib.sha1()
    for path in sorted(file_list):
        try:
            st = os.stat(path)
            record = f"{path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", errors="ignore")
        except OSError:
            record = f"{path}|missing".encode("utf-8", errors="ignore")
        hasher.update(record)
    return hasher.hexdigest()[:16]


def _cache_file_path(fingerprint: str, tile_size: int) -> str:
    return os.path.join(CACHE_DIR, f"{CACHE_VERSION}_{fingerprint}_{tile_size}.npz")


def _load_cached_tiles(fingerprint: str, tile_size: int):
    path = _cache_file_path(fingerprint, tile_size)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            tiles = data["tiles"]
            features = data["features"]
        if tiles.ndim != 4 or features.ndim != 2 or tiles.shape[0] != features.shape[0]:
            return None
        return tiles.astype(np.uint8, copy=False), features.astype(np.float32, copy=False)
    except Exception:
        return None


def _save_cached_tiles(fingerprint: str, tile_size: int, tiles: np.ndarray, features: np.ndarray):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_file_path(fingerprint, tile_size)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as file:
        np.savez_compressed(file, tiles=tiles, features=features)
    os.replace(tmp_path, path)


def _process_one_tile(args):
    path, tile_size = args
    try:
        img = read_image(path)
        img_small = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        return img_small
    except ImageLoadError:
        return None
    except Exception:
        return None


def prepare_tiles_parallel(
    file_list: List[str],
    tile_size: int,
    progress_callback: Callable[[float, str], None],
    dataset_fp: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cached = _load_cached_tiles(dataset_fp, tile_size)
    if cached is not None:
        progress_callback(0, f"Dùng cache tiles size {tile_size}px...")
        return cached

    tiles = []
    total = len(file_list)
    progress_callback(0, f"Đang nạp {total} ảnh mẫu (size {tile_size}px)...")

    tasks = [(path, tile_size) for path in file_list]
    max_workers = min(32, (os.cpu_count() or 4) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_one_tile, tasks))
        for i, res in enumerate(results):
            if res is not None:
                tiles.append(res)
            if i % 200 == 0:
                progress_callback((i / total) * 100, f"Loading: {i}/{total}")

    if not tiles:
        raise TileDatasetError("Không tìm thấy ảnh hợp lệ trong thư mục tiles!")

    tiles_arr = np.array(tiles, dtype=np.uint8)
    features_arr = _compute_features_batch(tiles_arr)
    _save_cached_tiles(dataset_fp, tile_size, tiles_arr, features_arr)
    return tiles_arr, features_arr


def resize_tiles_in_memory(
    base_tiles: np.ndarray,
    new_size: int,
    progress_callback: Callable[[float, str], None],
    dataset_fp: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cached = _load_cached_tiles(dataset_fp, new_size)
    if cached is not None:
        progress_callback(0, f"Dùng cache tiles size {new_size}px...")
        return cached

    count = len(base_tiles)
    progress_callback(0, f"Downscaling tiles về {new_size}px...")

    new_tiles = []
    for i, tile in enumerate(base_tiles):
        small = cv2.resize(tile, (new_size, new_size), interpolation=cv2.INTER_AREA)
        new_tiles.append(small)
        if i % 1000 == 0:
            progress_callback((i / count) * 100, f"Resizing: {i}/{count}")

    tiles_arr = np.array(new_tiles, dtype=np.uint8)
    features_arr = _compute_features_batch(tiles_arr)
    _save_cached_tiles(dataset_fp, new_size, tiles_arr, features_arr)
    return tiles_arr, features_arr


def multi_resolution_mosaic(
    target_path: str,
    tiles_folder: str,
    base_tile: int = 15,
    levels: int = 3,
    blend_factor: float = 0.2,
    progress_callback: Callable[[float, str], None] = lambda p, m: None,
    frame_callback=None,
    frame_every: int = 150,
) -> Tuple[np.ndarray, List[int]]:
    target = read_image(target_path)
    h_img, w_img = target.shape[:2]

    file_list = list_image_files(tiles_folder)
    if not file_list:
        raise TileDatasetError("Thư mục tiles trống!")

    sizes = level_sizes(base_tile, levels)
    max_size = sizes[0]
    min_size = sizes[-1]
    dataset_fp = _dataset_fingerprint(file_list)
    progress_callback(5, f"Levels cấu hình: {sizes}")

    tiles_db = {}
    base_t, base_f = prepare_tiles_parallel(file_list, max_size, progress_callback, dataset_fp)
    tiles_db[max_size] = (base_t, KDTreeNearestNeighbor(base_f))

    for sz in sizes[1:]:
        t_arr, f_arr = resize_tiles_in_memory(base_t, sz, progress_callback, dataset_fp)
        tiles_db[sz] = (t_arr, KDTreeNearestNeighbor(f_arr))

    progress_callback(30, "Đang ghép tranh (Adaptive Mode)...")
    mosaic = np.zeros_like(target)
    queue = deque()
    for y in range(0, h_img, max_size):
        for x in range(0, w_img, max_size):
            queue.append((x, y, max_size))

    total_pixels = h_img * w_img
    processed_pixels = 0
    blocks_count = 0
    last_ui_update = time.time()

    while queue:
        x, y, sz = queue.popleft()
        h_slice = min(sz, h_img - y)
        w_slice = min(sz, w_img - x)
        if h_slice <= 0 or w_slice <= 0:
            continue

        roi = target[y : y + h_slice, x : x + w_slice]
        mean, stddev = cv2.meanStdDev(roi)
        avg_std = np.mean(stddev)
        should_split = (sz > min_size) and (avg_std > DEFAULT_SPLIT_THRESHOLD)

        if should_split:
            half = sz // 2
            queue.append((x, y, half))
            queue.append((x + half, y, half))
            queue.append((x, y + half, half))
            queue.append((x + half, y + half, half))
        else:
            current_dataset = tiles_db.get(sz)
            if current_dataset is None:
                current_dataset = tiles_db[min_size]

            t_arr, tree = current_dataset
            query_vec = mean.flatten().astype(np.float32)
            idx_match = tree.query(query_vec)
            best_tile = t_arr[idx_match]

            if best_tile.shape[:2] != (h_slice, w_slice):
                tile_resized = cv2.resize(best_tile, (w_slice, h_slice))
                mosaic[y : y + h_slice, x : x + w_slice] = tile_resized
            else:
                mosaic[y : y + h_slice, x : x + w_slice] = best_tile[:h_slice, :w_slice]

            processed_pixels += h_slice * w_slice
            blocks_count += 1
            if blocks_count % frame_every == 0:
                pct = min(100, int(processed_pixels / total_pixels * 100))
                progress_callback(30 + (pct * 0.7), f"Rendering: {pct}%")
                if frame_callback:
                    now = time.time()
                    if now - last_ui_update > 0.033:
                        frame_callback(mosaic)
                        last_ui_update = now

    if frame_callback:
        frame_callback(mosaic)

    if blend_factor > 0:
        progress_callback(100, "Đang hòa trộn (Blending)...")
        mosaic = cv2.addWeighted(mosaic, 1.0 - blend_factor, target, blend_factor, 0)

    progress_callback(100, "Hoàn tất!")
    return mosaic, sizes