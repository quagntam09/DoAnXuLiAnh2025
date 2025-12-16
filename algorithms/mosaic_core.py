import cv2
import numpy as np
import os
import concurrent.futures
import shutil
from algorithms.kdtree_module import KDTree

def process_single_tile(args):
    """Hàm hỗ trợ xử lý 1 ảnh (để chạy đa luồng)"""
    path, size = args
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        
        # Ép nhỏ ảnh
        img_small = cv2.resize(img, (size, size))
        
        # Tính màu trung bình
        avg_color = np.mean(img_small, axis=(0,1))
        
        return (img_small, avg_color)
    except:
        return None

class MosaicGenerator:
    def __init__(self, target_path, tiles_folder, tile_size, blend_factor, scale_factor=1, use_adaptive=False):
        self.target_path = target_path
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.blend_factor = blend_factor
        self.scale_factor = scale_factor
        self.use_adaptive = use_adaptive

    def run(self, progress_callback):
        # BƯỚC 1: LOAD VÀ XỬ LÝ ẢNH TILES
        progress_callback(5, "Đang kiểm tra nguồn ảnh tiles...")
        
        tiles = []
        colors = []
        
        # Logic tự động chọn Flickr8k nếu folder nhập vào không hợp lệ hoặc rỗng
        use_fallback = False
        current_tiles_folder = self.tiles_folder
        
        # Kiểm tra folder người dùng chọn
        if not current_tiles_folder or not os.path.exists(current_tiles_folder) or len(os.listdir(current_tiles_folder)) == 0:
            use_fallback = True
        
        if use_fallback:
            # Thử tìm folder flickr8k mặc định ở thư mục gốc dự án
            default_flickr = os.path.join(os.getcwd(), "flickr8k")
            if os.path.exists(default_flickr) and len(os.listdir(default_flickr)) > 0:
                current_tiles_folder = default_flickr
                progress_callback(5, "Không tìm thấy ảnh tiles đầu vào, đang sử dụng bộ Flickr8k...")
            else:
                # Nếu cả 2 đều không được thì báo lỗi dựa trên input ban đầu của user
                if not os.path.exists(self.tiles_folder):
                     raise Exception(f"Thư mục không tồn tại: {self.tiles_folder}")
                else:
                     raise Exception("Folder ảnh phụ trống rỗng và không tìm thấy bộ Flickr8k!")

        # --- CACHING SYSTEM (Tăng tốc độ load ảnh) ---
        cache_filename = f"cache_v2_{self.tile_size}.npz"
        cache_path = os.path.join(current_tiles_folder, cache_filename)
        
        cache_loaded = False
        tiles_arr = None
        colors_list = None
        
        # Thử load cache
        if os.path.exists(cache_path):
            try:
                progress_callback(10, "Đang tải dữ liệu từ Cache (Siêu tốc)...")
                data = np.load(cache_path)
                tiles_arr = data['tiles']
                colors_list = data['colors'].tolist()
                cache_loaded = True
                progress_callback(30, f"Đã load {len(tiles_arr)} ảnh từ Cache!")
            except Exception as e:
                print(f"Lỗi load cache: {e}")
                cache_loaded = False

        if not cache_loaded:
            file_list = [f for f in os.listdir(current_tiles_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_files = len(file_list)

            progress_callback(10, f"Đang đọc và xử lý {total_files} ảnh (Lần đầu sẽ lâu)...")

            # --- SỬ DỤNG MULTITHREADING ---
            process_args = [
                (os.path.join(current_tiles_folder, f), self.tile_size) 
                for f in file_list
            ]

            completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_file = {executor.submit(process_single_tile, arg): arg for arg in process_args}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()
                    if result is not None:
                        img_small, avg_color = result
                        tiles.append(img_small)
                        colors.append(avg_color)
                    
                    completed_count += 1
                    if completed_count % 100 == 0:
                        percent = 10 + (completed_count / total_files) * 20
                        progress_callback(percent, f"Đang nén: {completed_count}/{total_files} ảnh")

            if not tiles: 
                raise Exception("Không tìm thấy ảnh hợp lệ nào!")
            
            tiles_arr = np.array(tiles)
            colors_list = [c.tolist() for c in colors]
            
            # Lưu cache để lần sau chạy nhanh hơn
            try:
                progress_callback(32, "Đang lưu Cache cho lần sau...")
                np.savez_compressed(cache_path, tiles=tiles_arr, colors=np.array(colors_list))
            except Exception as e:
                print(f"Không thể lưu cache: {e}")

        # BƯỚC 2: KHỞI TẠO KD-TREE (CUSTOM)
        progress_callback(35, "Đang xây dựng cây KD-Tree...")
        tree = KDTree(colors_list) 
        
        # Nếu dùng Adaptive, chuẩn bị thêm bộ tiles nhỏ
        tree_small = None
        tiles_small_arr = None
        small_size = self.tile_size // 2
        
        if self.use_adaptive:
            progress_callback(36, "Đang chuẩn bị chế độ Adaptive (Tạo tiles nhỏ)...")
            # Resize tiles lớn xuống nhỏ (nhanh hơn load lại)
            tiles_small_arr = np.array([cv2.resize(t, (small_size, small_size)) for t in tiles_arr])
            colors_small_list = [np.mean(t, axis=(0,1)).tolist() for t in tiles_small_arr]
            tree_small = KDTree(colors_small_list)

        # BƯỚC 3: XỬ LÝ ẢNH GỐC
        progress_callback(40, f"Đang xử lý ảnh gốc (Upscale {self.scale_factor}x)...")
        target = cv2.imread(self.target_path)
        if target is None: raise Exception("Lỗi đọc ảnh gốc")
        
        # Phóng to ảnh gốc nếu cần (để tăng độ chi tiết cho Mosaic)
        if self.scale_factor > 1:
            target = cv2.resize(target, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)

        h, w, _ = target.shape

        h_new = (h // self.tile_size) * self.tile_size
        w_new = (w // self.tile_size) * self.tile_size
        target = target[:h_new, :w_new] 

        mosaic = np.zeros_like(target)


        # BƯỚC 4: GHÉP ẢNH

        rows = range(0, h_new, self.tile_size)
        cols = range(0, w_new, self.tile_size)
        total_blocks = len(rows) * len(cols)
        processed_blocks = 0
        
        # Mảng lưu vết index ảnh đã dùng để check trùng lặp (chỉ dùng cho tiles lớn)
        num_rows = len(rows)
        num_cols = len(cols)
        used_indices = np.full((num_rows, num_cols), -1, dtype=int)

        progress_callback(45, f"Đang ghép (Adaptive: {self.use_adaptive})...")

        for r, y in enumerate(rows):
            for c, x in enumerate(cols):
                # Lấy vùng ảnh gốc
                roi = target[y:y+self.tile_size, x:x+self.tile_size]
                
                # Kiểm tra độ phức tạp của vùng ảnh (nếu Adaptive đang bật)
                is_complex = False
                if self.use_adaptive:
                    # Tính độ lệch chuẩn
                    (mean, std) = cv2.meanStdDev(roi)
                    avg_std = np.mean(std)
                    # Ngưỡng phức tạp (có thể tinh chỉnh, 18.0 là con số kinh nghiệm)
                    if avg_std > 18.0:
                        is_complex = True
                
                if is_complex and tree_small is not None:
                    # --- XỬ LÝ CHIA NHỎ (4 ô con) ---
                    # Thứ tự: [0,0], [0,1], [1,0], [1,1] (theo tọa độ sub-block)
                    sub_coords = [
                        (0, 0), (0, small_size),
                        (small_size, 0), (small_size, small_size)
                    ]
                    
                    for dy, dx in sub_coords:
                        # Vùng con trên ảnh gốc
                        sub_roi = roi[dy:dy+small_size, dx:dx+small_size]
                        sub_avg = np.mean(sub_roi, axis=(0,1)).tolist()
                        
                        # Tìm tile nhỏ tốt nhất
                        # Với tile nhỏ, ta ưu tiên khớp màu chính xác nhất (k=1) cho nhanh
                        idx_s, dist_s = tree_small.find_nearest(sub_avg)
                        
                        best_tile_s = tiles_small_arr[idx_s]
                        best_tile_avg_s = np.mean(best_tile_s, axis=(0,1)) # Tính lại hoặc lưu cache nếu cần
                        
                        # Color Correction cho ô nhỏ
                        diff_s = np.array(sub_avg) - np.array(best_tile_avg_s)
                        corrected_tile_s = np.clip(best_tile_s.astype(np.int16) + diff_s.astype(np.int16), 0, 255).astype(np.uint8)
                        
                        # Gán vào mosaic
                        mosaic[y+dy:y+dy+small_size, x+dx:x+dx+small_size] = corrected_tile_s
                    
                    # Đánh dấu ô lớn này là "đã dùng adaptive" (không lưu index cụ thể)
                    used_indices[r, c] = -2 

                else:
                    # --- XỬ LÝ BÌNH THƯỜNG (Ô Lớn) ---
                    # Tính màu trung bình và chuyển sang List
                    avg = np.mean(roi, axis=(0,1)).tolist() 

                    # Lấy 15 ứng viên tốt nhất thay vì chỉ 1
                    candidates = tree.find_k_nearest(avg, k=15)
                    
                    # Kiểm tra hàng xóm (Trái và Trên)
                    left_idx = used_indices[r, c-1] if c > 0 else -1
                    top_idx = used_indices[r-1, c] if r > 0 else -1
                    
                    chosen_idx = -1
                    
                    # Chọn ứng viên tốt nhất KHÔNG trùng với hàng xóm
                    for idx, dist in candidates:
                        if idx != left_idx and idx != top_idx:
                            chosen_idx = idx
                            break
                    
                    # Nếu tất cả đều trùng (rất hiếm), lấy cái tốt nhất
                    if chosen_idx == -1:
                        chosen_idx = candidates[0][0]
                    
                    # Lấy ảnh tile tốt nhất
                    best_tile = tiles_arr[chosen_idx]
                    best_tile_avg = colors_list[chosen_idx]

                    # --- COLOR CORRECTION (Hiệu chỉnh màu) ---
                    diff = np.array(avg) - np.array(best_tile_avg)
                    corrected_tile = np.clip(best_tile.astype(np.int16) + diff.astype(np.int16), 0, 255).astype(np.uint8)

                    # Lưu vết
                    used_indices[r, c] = chosen_idx

                    # Gán ảnh đã chỉnh màu vào mosaic
                    mosaic[y:y+self.tile_size, x:x+self.tile_size] = corrected_tile

                processed_blocks += 1
                if processed_blocks % 200 == 0:
                    percent = 45 + (processed_blocks / total_blocks) * 50
                    progress_callback(percent, f"Đang ghép: {int(percent)}%")

        # BƯỚC 5: BLENDING VÀ LƯU

        if self.blend_factor > 0:
            final_result = cv2.addWeighted(mosaic, 1.0 - self.blend_factor, target, self.blend_factor, 0)
        else:
            final_result = mosaic

        output_name = "result_mosaic_custom.jpg"
        cv2.imwrite(output_name, final_result)
        
        progress_callback(100, "Hoàn tất!")
        return output_name, final_result