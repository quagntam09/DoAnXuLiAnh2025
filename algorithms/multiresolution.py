import os
import cv2
import numpy as np
import time
from typing import Callable, List, Tuple

# Sửa import để chạy flat structure
from algorithms.average_color import extract
from algorithms.kdtree_nn import KDTreeNearestNeighbor

def level_sizes(base_tile: int, levels: int = 3) -> List[int]:
    if base_tile <= 0:
        raise ValueError("base_tile phải > 0")
    if levels <= 0:
        raise ValueError("levels phải > 0")

    sizes = []
    # Level đầu tiên tile to nhất -> Mosaic thô nhất
    s = base_tile * (2 ** (levels - 1))
    for _ in range(levels):
        sizes.append(int(s))
        s //= 2
        if s < base_tile: 
            break
    
    # Đảm bảo level cuối cùng chính xác là base_tile
    if sizes[-1] != base_tile:
        sizes.append(base_tile)
        
    # Loại bỏ trùng lặp và sort giảm dần
    sizes = sorted(list(set(sizes)), reverse=True)
    return sizes

def _list_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    return files

def prepare_tiles_for_level(file_list: List[str], tile_size: int,
                            progress_callback: Callable[[float, str], None] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc và resize ảnh tile về đúng kích thước tile_size.
    Tính màu trung bình cho từng tile.
    """
    if progress_callback is None:
        progress_callback = lambda p, msg: None

    tiles = []
    colors = []
    total = len(file_list)

    progress_callback(0, f"Đang xử lý tiles size {tile_size}px...")

    for i, path in enumerate(file_list):
        img = cv2.imread(path)
        if img is None:
            continue
        
        # Resize tile về kích thước grid hiện tại
        try:
            img_small = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        except Exception:
            continue

        tiles.append(img_small)
        colors.append(extract(img_small))  # (3,)

        if i % 200 == 0: # Update progress mỗi 200 ảnh để đỡ lag
            progress_callback((i / max(1, total)) * 100, f"Load tiles ({tile_size}px): {i}/{total}")

    if not tiles:
        raise Exception(f"Không tạo được tile nào ở size {tile_size}!")

    tiles_arr = np.array(tiles, dtype=np.uint8)
    colors_arr = np.array(colors, dtype=np.float32)
    return tiles_arr, colors_arr

def build_index(colors_arr: np.ndarray):
    return KDTreeNearestNeighbor(colors_arr)

def render_single_level(target_img, tiles_arr, index, tile_size,
                        progress_callback=None, frame_callback=None,
                        frame_every=200, max_fps=30):
    if progress_callback is None:
        progress_callback = lambda p, msg: None

    h, w = target_img.shape[:2]

    mosaic = np.zeros_like(target_img) 

    rows = range(0, h, tile_size)
    cols = range(0, w, tile_size)
    total_blocks = len(rows) * len(cols)
    done = 0

    min_dt = 1.0 / max(1, max_fps)
    last_push = 0.0

    for y in rows:
        for x in cols:
            # Cắt vùng ảnh gốc (ROI)
            h_slice = min(tile_size, h - y)
            w_slice = min(tile_size, w - x)
            
            roi = target_img[y:y + h_slice, x:x + w_slice]
            
            # Nếu ROI nhỏ hơn tile_size (ở biên), cần padding hoặc resize tạm để tính avg
            avg = extract(roi)
            
            # Tìm tile gần nhất bằng KD-Tree
            idx = index.query(avg)
            
            # Lấy tile kết quả và crop nếu ở biên
            tile_match = tiles_arr[idx]
            mosaic[y:y + h_slice, x:x + w_slice] = tile_match[:h_slice, :w_slice]

            done += 1

            # Update progress UI
            if done % 100 == 0:
                progress_callback((done / max(1, total_blocks)) * 100,
                                    f"Ghép level {tile_size}px: {done}/{total_blocks}")

            # Update Live Preview
            if frame_callback is not None and (done % max(1, frame_every) == 0):
                now = time.time()
                if (now - last_push) >= min_dt:
                    last_push = now
                    frame_callback(mosaic) # Gửi ảnh về GUI

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
        raise Exception("Không đọc được ảnh gốc!")

    progress_callback(1, "Đang quét danh sách file tiles...")
    file_list = _list_image_files(tiles_folder)
    if not file_list:
        raise Exception("Thư mục tiles trống!")
    
    sizes = level_sizes(base_tile, levels)
    progress_callback(2, f"Multi-res levels: {sizes}")

    final_mosaic = target.copy() # Khởi tạo bằng ảnh gốc

    for li, ts in enumerate(sizes):
        # Crop ảnh gốc cho chẵn với tile_size (để xử lý dễ hơn)
        h, w = target.shape[:2]
        h_new = (h // ts) * ts
        w_new = (w // ts) * ts
        # Nếu ảnh quá nhỏ so với tile, bỏ qua level này
        if h_new == 0 or w_new == 0:
            continue
            
        target_crop = target[:h_new, :w_new]

        # 3.1 Chuẩn bị tiles
        tiles_arr, colors_arr = prepare_tiles_for_level(file_list, ts, progress_callback)

        # 3.2 Build KD-tree
        progress_callback(20, f"Build KD-tree level {ts}...")
        index = build_index(colors_arr)

        # 3.3 Render mosaic
        progress_callback(30, f"Render mosaic level {ts}...")
        mosaic_lv = render_single_level(
            target_crop,
            tiles_arr,
            index,
            ts,
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            frame_every=frame_every
        )

        final_mosaic = mosaic_lv
        progress_callback(100, f"Hoàn tất level {ts}px")
        
        time.sleep(0.5)

    # --- 4. Blend với ảnh gốc ---
    if blend_factor > 0:
        progress_callback(95, "Blending với ảnh gốc...")
        # Resize ảnh gốc khớp với final_mosaic hiện tại (vì mosaic có thể bị crop chút xíu)
        mh, mw = final_mosaic.shape[:2]
        target_resized = cv2.resize(target, (mw, mh))
        
        final_mosaic = cv2.addWeighted(
            final_mosaic,
            1.0 - blend_factor,
            target_resized,
            blend_factor,
            0
        )

    progress_callback(100, "Xong!")
    return final_mosaic, sizes