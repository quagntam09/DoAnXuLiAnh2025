import os
import cv2
import numpy as np
import time
from typing import Callable, List, Tuple, Dict
import concurrent.futures
from collections import deque

# Import thuật toán lõi
from algorithms.average_color import extract
from algorithms.kdtree_nn import KDTreeNearestNeighbor

# --- CẤU HÌNH ---
# Trọng số cho thành phần Texture (StdDev) khi query KD-Tree.
# < 1.0: Ưu tiên đúng màu hơn đúng vân.
# > 1.0: Ưu tiên đúng vân hơn đúng màu.
TEXTURE_WEIGHT = 0.8 

def level_sizes(base_tile: int, levels: int = 3) -> List[int]:
    """
    Tính toán danh sách các kích thước tile cần chuẩn bị.
    Đảm bảo tuân thủ quy tắc lũy thừa 2 để khớp với Quadtree.
    Ví dụ base=10, levels=3 -> [40, 20, 10]
    """
    if base_tile <= 0: raise ValueError("base_tile phải > 0")
    if levels <= 0: raise ValueError("levels phải > 0")

    sizes = []
    # Size lớn nhất
    max_s = base_tile * (2 ** (levels - 1))
    
    current = max_s
    while current >= base_tile:
        sizes.append(current)
        current //= 2
        
    return sizes

def _list_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    return files

def _process_one_tile(args):
    """Xử lý 1 tile: Đọc -> Resize -> Trích xuất đặc trưng (Màu + Texture)"""
    path, tile_size = args
    try:
        img = cv2.imread(path)
        if img is None: return None
        
        # Resize về kích thước cần thiết
        img_small = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        
        # Trích xuất đặc trưng (Mean + Std nếu file extract hỗ trợ)
        feature_vec = extract(img_small) 
        
        # Nếu feature có chứa texture (vector 6 chiều), ta scale phần texture
        # để điều chỉnh độ ưu tiên
        if feature_vec.shape[0] == 6:
            feature_vec[3:] *= TEXTURE_WEIGHT
            
        return img_small, feature_vec
    except Exception:
        return None

def prepare_tiles_parallel(file_list: List[str], tile_size: int,
                            progress_callback: Callable[[float, str], None]) -> Tuple[np.ndarray, np.ndarray]:
    """Load và xử lý tiles ban đầu (kích thước lớn nhất) bằng đa luồng."""
    tiles = []
    features = []
    total = len(file_list)

    progress_callback(0, f"Đang nạp {total} ảnh mẫu (size {tile_size}px)...")

    tasks = [(path, tile_size) for path in file_list]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_one_tile, tasks))

        for i, res in enumerate(results):
            if res is not None:
                tiles.append(res[0])
                features.append(res[1])
            
            if i % 200 == 0:
                progress_callback((i / total) * 100, f"Loading: {i}/{total}")

    if not tiles:
        raise Exception(f"Không tìm thấy ảnh hợp lệ trong thư mục tiles!")

    return np.array(tiles, dtype=np.uint8), np.array(features, dtype=np.float32)

def resize_tiles_in_memory(base_tiles: np.ndarray, new_size: int, 
                           progress_callback: Callable[[float, str], None]) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo tiles kích thước nhỏ hơn từ tiles đã load trên RAM."""
    count = len(base_tiles)
    progress_callback(0, f"Downscaling tiles về {new_size}px...")
    
    new_tiles = []
    new_features = []
    
    for i, tile in enumerate(base_tiles):
        small = cv2.resize(tile, (new_size, new_size), interpolation=cv2.INTER_AREA)
        new_tiles.append(small)
        
        # Trích xuất lại đặc trưng ở size mới (quan trọng vì texture thay đổi theo size)
        feat = extract(small)
        if feat.shape[0] == 6:
            feat[3:] *= TEXTURE_WEIGHT
        new_features.append(feat)
        
        if i % 1000 == 0:
             progress_callback((i / count) * 100, f"Resizing: {i}/{count}")
             
    return np.array(new_tiles, dtype=np.uint8), np.array(new_features, dtype=np.float32)

def multi_resolution_mosaic(
    target_path: str,
    tiles_folder: str,
    base_tile: int = 15,
    levels: int = 3,
    blend_factor: float = 0.2,
    progress_callback: Callable[[float, str], None] = lambda p, m: None,
    frame_callback=None,
    frame_every: int = 150
) -> Tuple[np.ndarray, List[int]]:
    
    # 1. Cấu hình Quadtree
    # Ngưỡng chia cắt: Nếu độ lệch chuẩn trung bình vùng ảnh > ngưỡng này -> chia nhỏ
    SPLIT_THRESHOLD = 20.0 

    # --- SETUP DỮ LIỆU ---
    target = cv2.imread(target_path)
    if target is None: raise Exception("Lỗi đọc ảnh gốc!")
    h_img, w_img = target.shape[:2]

    file_list = _list_image_files(tiles_folder)
    if not file_list: raise Exception("Thư mục tiles trống!")
    
    # Tính toán các level kích thước: [Max, ..., Min]
    sizes = level_sizes(base_tile, levels)
    max_size = sizes[0]
    min_size = sizes[-1]
    
    progress_callback(5, f"Levels cấu hình: {sizes}")

    # --- LOAD & PREPARE TILES ---
    # Database lưu tiles theo kích thước: {size: (tiles_array, kdtree)}
    tiles_db = {}

    # 1. Load tiles ở size lớn nhất
    base_t, base_f = prepare_tiles_parallel(file_list, max_size, progress_callback)
    tiles_db[max_size] = (base_t, KDTreeNearestNeighbor(base_f))

    # 2. Downscale cho các size nhỏ hơn
    for sz in sizes[1:]:
        t_arr, f_arr = resize_tiles_in_memory(base_t, sz, progress_callback)
        tiles_db[sz] = (t_arr, KDTreeNearestNeighbor(f_arr))

    # --- QUADTREE PROCESS ---
    progress_callback(30, "Đang ghép tranh (Adaptive Mode)...")
    mosaic = np.zeros_like(target)
    
    # Queue chứa (x, y, size). Bắt đầu với lưới lớn nhất.
    queue = deque()
    for y in range(0, h_img, max_size):
        for x in range(0, w_img, max_size):
            queue.append((x, y, max_size))

    total_pixels = h_img * w_img
    processed_pixels = 0
    blocks_count = 0
    
    last_ui_update = time.time()

    while queue:
        x, y, sz = queue.popleft() # Dùng popleft (BFS) để hiển thị dần từ thô đến tinh
        
        # Xử lý biên ảnh
        h_slice = min(sz, h_img - y)
        w_slice = min(sz, w_img - x)
        if h_slice <= 0 or w_slice <= 0: continue

        roi = target[y:y+h_slice, x:x+w_slice]
        
        # Tính toán thống kê màu và texture của vùng ảnh đích (ROI)
        # mean: (3,1), stddev: (3,1)
        mean, stddev = cv2.meanStdDev(roi)
        avg_std = np.mean(stddev)
        
        # QUYẾT ĐỊNH: Chia nhỏ hay Dừng?
        # Chia nhỏ nếu: Chưa đạt size nhỏ nhất VÀ độ phức tạp chi tiết > ngưỡng
        should_split = (sz > min_size) and (avg_std > SPLIT_THRESHOLD)

        if should_split:
            # Chia làm 4 ô con
            half = sz // 2
            queue.append((x, y, half))
            queue.append((x + half, y, half))
            queue.append((x, y + half, half))
            queue.append((x + half, y + half, half))
        else:
            # TÌM ẢNH GHÉP (MATCHING)
            # Lấy bộ dataset phù hợp kích thước
            current_dataset = tiles_db.get(sz)
            
            # Fallback nếu không có size chính xác (do biên ảnh lẻ), dùng size nhỏ nhất
            if current_dataset is None:
                current_dataset = tiles_db[min_size] 

            t_arr, tree = current_dataset
            
            # Tạo vector truy vấn (Query Vector)
            # Tận dụng luôn kết quả mean, stddev vừa tính để ko phải gọi hàm extract() lại -> TỐI ƯU TỐC ĐỘ
            mean_flat = mean.flatten().astype(np.float32)
            
            # Kiểm tra xem KD-Tree đang dùng 3 chiều (chỉ màu) hay 6 chiều (màu + texture)
            if tree.colors.shape[1] == 6:
                std_flat = stddev.flatten().astype(np.float32) * TEXTURE_WEIGHT
                query_vec = np.concatenate([mean_flat, std_flat])
            else:
                query_vec = mean_flat

            # Query KD-Tree
            idx_match = tree.query(query_vec)
            best_tile = t_arr[idx_match]

            # Gán vào ảnh kết quả (cắt nếu ở biên)
            if best_tile.shape[:2] != (h_slice, w_slice):
                # Resize nhẹ nếu kích thước không khớp hoàn toàn (do fallback)
                tile_resized = cv2.resize(best_tile, (w_slice, h_slice))
                mosaic[y:y+h_slice, x:x+w_slice] = tile_resized
            else:
                mosaic[y:y+h_slice, x:x+w_slice] = best_tile[:h_slice, :w_slice]

            # Cập nhật tiến độ
            processed_pixels += (h_slice * w_slice)
            blocks_count += 1
            
            if blocks_count % frame_every == 0:
                pct = min(100, int(processed_pixels / total_pixels * 100))
                progress_callback(30 + (pct * 0.7), f"Rendering: {pct}%")
                
                # Cập nhật preview (giới hạn 30fps)
                if frame_callback:
                    now = time.time()
                    if now - last_ui_update > 0.033:
                        frame_callback(mosaic)
                        last_ui_update = now

    # Final preview update
    if frame_callback: frame_callback(mosaic)
    
    # --- BLENDING ---
    if blend_factor > 0:
        progress_callback(100, "Đang hòa trộn (Blending)...")
        # Overlay ảnh gốc mờ lên trên mosaic để làm mềm các cạnh
        mosaic = cv2.addWeighted(mosaic, 1.0 - blend_factor, target, blend_factor, 0)

    progress_callback(100, "Hoàn tất!")
    return mosaic, sizes