import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import core thuật toán
from algorithms.mosaic_core import MosaicGenerator

# --- HÀM HỖ TRỢ HIỂN THỊ ẢNH ---
def bgr_to_tk(img_bgr: np.ndarray, max_w=800, max_h=800) -> ImageTk.PhotoImage:
    """Chuyển đổi ảnh OpenCV (BGR) sang ảnh Tkinter để hiển thị, có resize giữ tỉ lệ."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Tính tỉ lệ scale để fit vào khung hình
    scale = min(max_w / w, max_h / h, 1.0)
    
    # Chỉ resize nếu ảnh lớn hơn khung
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần Mềm Tạo Tranh Mosaic Nghệ Thuật")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Style theme
        style = ttk.Style(self)
        style.theme_use('clam') # Hoặc 'alt', 'default' tùy OS
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#333")

        # --- Variables ---
        self.target_path = tk.StringVar(value="")
        self.tiles_folder = tk.StringVar(value="")
        
        # Giá trị mặc định
        self.tile_size = tk.IntVar(value=15)
        self.levels = tk.IntVar(value=3)
        self.blend = tk.DoubleVar(value=0.2)
        
        self._current_img = None  # Lưu ảnh gốc
        self._result_img = None   # Lưu ảnh kết quả
        self._photo = None        # Giữ reference cho Tkinter khỏi bị garbage collect

        self._build_ui()

    def _build_ui(self):
        # Layout chính: Trái (Controls) - Phải (Preview)
        main_paned = tk.PanedWindow(self, orient="horizontal", sashwidth=5, bg="#dcdcdc")
        main_paned.pack(fill="both", expand=True)

        # === PANEL TRÁI: ĐIỀU KHIỂN ===
        left_frame = ttk.Frame(main_paned, padding=15)
        main_paned.add(left_frame, minsize=350, width=380)

        # 1. Logo / Header
        lbl_title = ttk.Label(left_frame, text="🛠 BẢNG ĐIỀU KHIỂN", style="Header.TLabel")
        lbl_title.pack(anchor="w", pady=(0, 15))

        # 2. Bước 1: Chọn dữ liệu
        grp_input = ttk.LabelFrame(left_frame, text="1. Chọn Dữ Liệu", padding=10)
        grp_input.pack(fill="x", pady=5)

        # Nút chọn ảnh gốc
        ttk.Label(grp_input, text="Ảnh gốc (Chủ đề):").pack(anchor="w")
        btn_target = ttk.Button(grp_input, text="📂 Mở ảnh gốc...", command=self.pick_target)
        btn_target.pack(fill="x", pady=(2, 8))
        self.lbl_target_name = ttk.Label(grp_input, text="(Chưa chọn ảnh)", foreground="gray", wraplength=300)
        self.lbl_target_name.pack(anchor="w", pady=(0, 10))

        # Nút chọn folder tiles
        ttk.Label(grp_input, text="Kho ảnh ghép (Dataset):").pack(anchor="w")
        btn_tiles = ttk.Button(grp_input, text="📂 Chọn thư mục ảnh nhỏ...", command=self.pick_tiles_folder)
        btn_tiles.pack(fill="x", pady=(2, 8))
        self.lbl_tiles_name = ttk.Label(grp_input, text="(Chưa chọn thư mục)", foreground="gray", wraplength=300)
        self.lbl_tiles_name.pack(anchor="w")

        # 3. Bước 2: Cấu hình thuật toán
        grp_config = ttk.LabelFrame(left_frame, text="2. Tùy Chỉnh Nghệ Thuật", padding=10)
        grp_config.pack(fill="x", pady=15)

        # Slider: Kích thước ô
        self.lbl_tile_val = ttk.Label(grp_config, text=f"Kích thước ô nhỏ: {self.tile_size.get()} px")
        self.lbl_tile_val.pack(anchor="w")
        scale_tile = ttk.Scale(grp_config, from_=5, to=80, variable=self.tile_size, 
                               command=lambda v: self.lbl_tile_val.config(text=f"Kích thước ô nhỏ: {int(float(v))} px"))
        scale_tile.pack(fill="x", pady=(0, 10))

        # Slider: Độ chi tiết (Levels)
        self.lbl_level_val = ttk.Label(grp_config, text=f"Độ phân giải (Levels): {self.levels.get()}")
        self.lbl_level_val.pack(anchor="w")
        scale_level = ttk.Scale(grp_config, from_=1, to=6, variable=self.levels,
                                command=lambda v: self.lbl_level_val.config(text=f"Độ phân giải (Levels): {int(float(v))}"))
        scale_level.pack(fill="x", pady=(0, 10))

        # Slider: Pha trộn
        self.lbl_blend_val = ttk.Label(grp_config, text=f"Pha trộn ảnh gốc: {int(self.blend.get()*100)}%")
        self.lbl_blend_val.pack(anchor="w")
        scale_blend = ttk.Scale(grp_config, from_=0.0, to=1.0, variable=self.blend,
                                command=lambda v: self.lbl_blend_val.config(text=f"Pha trộn ảnh gốc: {int(float(v)*100)}%"))
        scale_blend.pack(fill="x")
        ttk.Label(grp_config, text="(Kéo cao để ảnh rõ nét hơn, thấp để nghệ thuật hơn)", 
                  font=("Arial", 8, "italic"), foreground="gray").pack(anchor="w")

        # 4. Bước 3: Hành động
        grp_action = ttk.LabelFrame(left_frame, text="3. Thực Hiện", padding=10)
        grp_action.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(grp_action, text="▶ BẮT ĐẦU TẠO TRANH", command=self.run_mosaic)
        self.btn_run.pack(fill="x", pady=5)
        
        self.progress = ttk.Progressbar(grp_action, mode="determinate")
        self.progress.pack(fill="x", pady=5)
        
        self.status = tk.StringVar(value="Sẵn sàng.")
        self.lbl_status = ttk.Label(grp_action, textvariable=self.status, foreground="blue", wraplength=300)
        self.lbl_status.pack(fill="x")

        # Nút Lưu (nằm riêng)
        self.btn_save = ttk.Button(left_frame, text="💾 Lưu Kết Quả Về Máy", command=self.save_as, state="disabled")
        self.btn_save.pack(fill="x", pady=20, side="bottom")

        # === PANEL PHẢI: PREVIEW ===
        right_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(right_frame)

        # Khung chứa ảnh (Canvas hoặc Label)
        self.preview_container = tk.Label(right_frame, bg="#333333", text="Khu vực hiển thị ảnh", fg="white")
        self.preview_container.pack(fill="both", expand=True)

    # --- LOGIC XỬ LÝ ---

    def pick_target(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh gốc",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        
        # Reset
        self.target_path.set(path)
        self.lbl_target_name.config(text=f"✔ {os.path.basename(path)}", foreground="green")
        
        # Load & Show
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", "File ảnh bị lỗi hoặc không hỗ trợ!")
            return

        self._current_img = img
        self._result_img = None
        self.btn_save.config(state="disabled")
        self.show_image(img)
        self.status.set(f"Đã tải ảnh gốc.")

    def pick_tiles_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa tập ảnh nhỏ")
        if folder:
            self.tiles_folder.set(folder)
            self.lbl_tiles_name.config(text=f"✔ .../{os.path.basename(folder)}", foreground="green")
            self.status.set("Đã chọn kho ảnh mẫu.")

    def show_image(self, img_bgr: np.ndarray):
        # Lấy kích thước thực tế của khung hiển thị để resize cho vừa vặn
        w = self.preview_container.winfo_width()
        h = self.preview_container.winfo_height()
        if w < 100: w = 800 # Fallback khi chưa render xong
        if h < 100: h = 600

        self._photo = bgr_to_tk(img_bgr, max_w=w, max_h=h)
        self.preview_container.configure(image=self._photo, text="")

    def run_mosaic(self):
        target = self.target_path.get().strip()
        tiles = self.tiles_folder.get().strip()

        if not target or not os.path.exists(target):
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn 'Ảnh gốc' trước!")
            return
        if not tiles or not os.path.isdir(tiles):
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn 'Kho ảnh ghép' trước!")
            return

        # Khóa giao diện
        self.btn_run.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.progress["value"] = 0
        self.status.set("Đang khởi động thuật toán...")

        # Params
        t_size = int(self.tile_size.get())
        levs = int(self.levels.get())
        bl = float(self.blend.get())

        # Callbacks cập nhật UI từ Thread
        def on_progress(p, msg):
            self.after(0, lambda: self.progress.configure(value=float(p)))
            self.after(0, lambda: self.status.set(msg))

        def on_frame(frame_img):
            # Copy để tránh conflict memory khi đang render
            show_img = frame_img.copy()
            self.after(0, lambda: self.show_image(show_img))

        def worker_thread():
            try:
                gen = MosaicGenerator(
                    target_path=target,
                    tiles_folder=tiles,
                    tile_size=t_size,
                    blend_factor=bl,
                    levels=levs,
                    frame_every=150 # Cập nhật preview mượt hơn
                )
                
                # Chạy thuật toán
                final_img = gen.run(progress_callback=on_progress, frame_callback=on_frame)
                
                # Hoàn tất
                self._result_img = final_img
                self.after(0, lambda: self.show_image(final_img))
                self.after(0, lambda: self.btn_save.config(state="normal"))
                self.after(0, lambda: messagebox.showinfo("Hoàn tất", "Đã tạo tranh Mosaic thành công!"))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda err=str(e): messagebox.showerror("Lỗi Runtime", f"Có lỗi xảy ra:\n{err}"))
            finally:
                self.after(0, lambda: self.btn_run.config(state="normal"))
                self.after(0, lambda: self.status.set("Đã xong."))

        threading.Thread(target=worker_thread, daemon=True).start()

    def save_as(self):
        if self._result_img is None:
            return
        path = filedialog.asksaveasfilename(
            title="Lưu tác phẩm",
            defaultextension=".jpg",
            filetypes=[("JPG Image", "*.jpg"), ("PNG Image", "*.png")]
        )
        if path:
            success, buf = cv2.imencode(os.path.splitext(path)[1], self._result_img)
            if success:
                with open(path, "wb") as f:
                    buf.tofile(f)
                messagebox.showinfo("Đã lưu", f"Ảnh đã được lưu tại:\n{path}")
            else:
                messagebox.showerror("Lỗi", "Không thể lưu file.")

if __name__ == "__main__":
    app = App()
    app.mainloop()