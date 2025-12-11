import cv2
import numpy as np
import os
from algorithms.kdtree_module import KDTree

class MosaicGenerator:
    def __init__(self, target_path, tiles_folder, tile_size, blend_factor):
        self.target_path = target_path
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.blend_factor = blend_factor

    def run(self, progress_callback):
        # --- BƯỚC 1: LOAD TILES ---
        progress_callback(5, "Đang đọc và ép nhỏ các ảnh tiles...")
        
        tiles = []
        colors = []
        file_list = os.listdir(self.tiles_folder)
        total_files = len(file_list)

        if total_files == 0: raise Exception("Folder ảnh phụ trống rỗng!")

        for i, filename in enumerate(file_list):
            path = os.path.join(self.tiles_folder, filename)
            try:
                img = cv2.imread(path)
                if img is not None:
                    # Ép nhỏ ảnh ngay lập tức để tiết kiệm RAM
                    img_small = cv2.resize(img, (self.tile_size, self.tile_size))
                    tiles.append(img_small)
                    # Tính màu trung bình
                    avg_color = np.mean(img_small, axis=(0,1))
                    colors.append(avg_color)
            except:
                pass
            
            if i % 50 == 0:
                percent = 5 + (i / total_files) * 25 # Từ 5% đến 30%
                progress_callback(percent, f"Đang nén: {i}/{total_files} ảnh")

        if not tiles: raise Exception("Không tìm thấy ảnh hợp lệ!")
        
        tiles_arr = np.array(tiles)
        colors_arr = np.array(colors)

        # --- BƯỚC 2: XÂY DỰNG KD-TREE ---
        progress_callback(35, "Đang lập chỉ mục KD-Tree...")
        tree = KDTree(colors_arr)

        # --- BƯỚC 3: XỬ LÝ ẢNH GỐC ---
        progress_callback(40, "Đang xử lý ảnh gốc...")
        target = cv2.imread(self.target_path)
        if target is None: raise Exception("Lỗi đọc ảnh gốc")

        h, w, _ = target.shape
        h_new = (h // self.tile_size) * self.tile_size
        w_new = (w // self.tile_size) * self.tile_size
        target = target[:h_new, :w_new] # Crop cho chẵn

        mosaic = np.zeros_like(target)

        # --- BƯỚC 4: GHÉP ẢNH ---
        rows = range(0, h_new, self.tile_size)
        cols = range(0, w_new, self.tile_size)
        total_blocks = len(rows) * len(cols)
        processed_blocks = 0

        progress_callback(45, "Đang ghép hàng nghìn ảnh nhỏ...")

        for y in rows:
            for x in cols:
                # Lấy vùng ảnh gốc
                roi = target[y:y+self.tile_size, x:x+self.tile_size]
                avg = np.mean(roi, axis=(0,1))

                # --- GỌI KD-TREE TÌM KIẾM ---
                _, idx = tree.query(avg)
                
                # Dán ảnh tìm được vào
                mosaic[y:y+self.tile_size, x:x+self.tile_size] = tiles_arr[idx]

                processed_blocks += 1
                if processed_blocks % 200 == 0:
                    percent = 45 + (processed_blocks / total_blocks) * 50
                    progress_callback(percent, f"Đang ghép: {int(percent)}%")

        # --- BƯỚC 5: BLENDING & SAVE ---
        if self.blend_factor > 0:
            progress_callback(98, "Đang phủ màu (Blending)...")
            final_result = cv2.addWeighted(mosaic, 1.0 - self.blend_factor, target, self.blend_factor, 0)
        else:
            final_result = mosaic

        output_name = "result_mosaic_hq.jpg"
        cv2.imwrite(output_name, final_result)
        
        progress_callback(100, "Hoàn tất!")
        
        # Trả về đường dẫn file và object ảnh để hiển thị
        return output_name, final_result