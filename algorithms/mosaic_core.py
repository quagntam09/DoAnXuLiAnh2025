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
        # BƯỚC 1: LOAD VÀ XỬ LÝ ẢNH TILES
        progress_callback(5, "Đang đọc và ép nhỏ các ảnh tiles...")
        
        tiles = []
        colors = []
        
        if not os.path.exists(self.tiles_folder):
            raise Exception(f"Thư mục không tồn tại: {self.tiles_folder}")
            
        file_list = os.listdir(self.tiles_folder)
        total_files = len(file_list)

        if total_files == 0: 
            raise Exception("Folder ảnh phụ trống rỗng!")

        for i, filename in enumerate(file_list):
            path = os.path.join(self.tiles_folder, filename)
            try:
                img = cv2.imread(path)
                
                if img is not None:
                    # Ép nhỏ ảnh
                    img_small = cv2.resize(img, (self.tile_size, self.tile_size))
                    tiles.append(img_small)
                    
                    # Tính màu trung bình
                    avg_color = np.mean(img_small, axis=(0,1))
                    colors.append(avg_color)
            except:
                pass
            
            if i % 50 == 0:
                percent = 5 + (i / total_files) * 25 
                progress_callback(percent, f"Đang nén: {i}/{total_files} ảnh")

        if not tiles: 
            raise Exception("Không tìm thấy ảnh hợp lệ nào!")
        
        tiles_arr = np.array(tiles)
        
        colors_list = [c.tolist() for c in colors] 

        # BƯỚC 2: KHỞI TẠO KD-TREE (CUSTOM)
        progress_callback(35, "Đang xây dựng cây KD-Tree (Tự viết)...")
        tree = KDTree(colors_list) 

        # BƯỚC 3: XỬ LÝ ẢNH GỐC
        progress_callback(40, "Đang xử lý ảnh gốc...")
        target = cv2.imread(self.target_path)
        if target is None: raise Exception("Lỗi đọc ảnh gốc")

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

        progress_callback(45, "Đang ghép (Custom Algorithm)...")

        for y in rows:
            for x in cols:
                # Lấy vùng ảnh gốc
                roi = target[y:y+self.tile_size, x:x+self.tile_size]
                
                # Tính màu trung bình và chuyển sang List
                avg = np.mean(roi, axis=(0,1)).tolist() 

                # Hàm find_nearest trả về (index, distance)
                idx, dist = tree.find_nearest(avg)
                
                # Lấy ảnh tile từ mảng dựa trên index tìm được
                mosaic[y:y+self.tile_size, x:x+self.tile_size] = tiles_arr[idx]

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