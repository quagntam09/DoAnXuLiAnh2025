import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import cv2
import shutil

from algorithms.mosaic_core import MosaicGenerator

class PhotomosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Mosaic - MVC Separation Architecture")
        self.root.geometry("1200x850")
        
        self.target_image_path = None
        self.tiles_folder_path = None
        self.tk_image_display = None 
        self.current_result_path = None # Đường dẫn ảnh kết quả tạm thời
        
        self._setup_ui()

    def _setup_ui(self):
        # --- KHUNG TRÁI ---
        control_frame = tk.Frame(self.root, width=320, bg="#f5f5f5", padx=20, pady=20)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(control_frame, text="BẢNG ĐIỀU KHIỂN", font=("Segoe UI", 14, "bold"), bg="#f5f5f5").pack(pady=(0, 20))

        # 1. Input Ảnh
        tk.Label(control_frame, text="1. Ảnh Gốc:", bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        tk.Button(control_frame, text="Chọn Ảnh Gốc", command=self.select_target_image, bg="white").pack(fill=tk.X, pady=5)
        self.lbl_img_path = tk.Label(control_frame, text="...", fg="gray", bg="#f5f5f5", wraplength=280)
        self.lbl_img_path.pack(fill=tk.X, pady=(0, 15))

        # 2. Input Folder
        tk.Label(control_frame, text="2. Folder Tiles:", bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        tk.Button(control_frame, text="Chọn Folder Tiles", command=self.select_tile_folder, bg="white").pack(fill=tk.X, pady=5)
        self.lbl_folder_path = tk.Label(control_frame, text="...", fg="gray", bg="#f5f5f5", wraplength=280)
        self.lbl_folder_path.pack(fill=tk.X, pady=(0, 15))

        # 3. Thông số
        tk.Label(control_frame, text="3. Kích thước Tile (px):", bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        self.entry_tile_size = tk.Entry(control_frame, font=("Segoe UI", 11))
        self.entry_tile_size.insert(0, "20")
        self.entry_tile_size.pack(fill=tk.X, pady=5)

        tk.Label(control_frame, text="4. Blending (0.0 - 0.8):", bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(fill=tk.X, pady=(15, 0))
        self.slider_blend = tk.Scale(control_frame, from_=0.0, to=0.8, resolution=0.05, orient=tk.HORIZONTAL, bg="#f5f5f5")
        self.slider_blend.set(0.2)
        self.slider_blend.pack(fill=tk.X)

        # 5. Upscale
        tk.Label(control_frame, text="5. Độ phân giải ảnh ra (Upscale):", bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(fill=tk.X, pady=(15, 0))
        self.combo_scale = ttk.Combobox(control_frame, values=["1x (Mặc định)", "2x (Chi tiết)", "3x (Rất nét)", "4x (Siêu nét)"], state="readonly")
        self.combo_scale.current(1) # Default 2x
        self.combo_scale.pack(fill=tk.X, pady=5)

        # 6. Adaptive Tiling
        self.var_adaptive = tk.BooleanVar(value=False)
        self.chk_adaptive = tk.Checkbutton(control_frame, text="6. Chế độ thông minh (Adaptive)", variable=self.var_adaptive, bg="#f5f5f5", font=("Segoe UI", 10))
        self.chk_adaptive.pack(fill=tk.X, pady=(15, 0))

        # Nút Chạy
        self.btn_run = tk.Button(control_frame, text="TẠO ẢNH (Logic Separate)", command=self.on_click_run, 
                                 bg="#28a745", fg="white", font=("Segoe UI", 12, "bold"), height=2)
        self.btn_run.pack(fill=tk.X, pady=(30, 10))

        # Nút Lưu (Mới)
        self.btn_save = tk.Button(control_frame, text="LƯU ẢNH KẾT QUẢ", command=self.save_image,
                                  bg="#007bff", fg="white", font=("Segoe UI", 11, "bold"), height=2, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=(0, 20))
        
        # Progress
        self.lbl_status = tk.Label(control_frame, text="Sẵn sàng", bg="#f5f5f5", fg="#007bff")
        self.lbl_status.pack(side=tk.BOTTOM, pady=5)
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # --- KHUNG PHẢI (Display) ---
        display_frame = tk.Frame(self.root, bg="#222")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas_label = tk.Label(display_frame, text="Ảnh kết quả sẽ hiện ở đây", bg="#222", fg="#888")
        self.canvas_label.pack(expand=True)

    # --- UI EVENT HANDLERS ---
    def select_target_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.target_image_path = path
            self.lbl_img_path.config(text=os.path.basename(path))
            self.show_image_preview(path)

    def select_tile_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.tiles_folder_path = path
            self.lbl_folder_path.config(text=os.path.basename(path))

    def save_image(self):
        if not self.current_result_path or not os.path.exists(self.current_result_path):
            messagebox.showerror("Lỗi", "Chưa có ảnh kết quả để lưu!")
            return
            
        # Mở hộp thoại chọn nơi lưu
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
            title="Lưu ảnh Mosaic"
        )
        
        if file_path:
            try:
                shutil.copy(self.current_result_path, file_path)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi lưu file", str(e))

    def show_image_preview(self, img_source):
        # Xử lý hiển thị ảnh (từ path hoặc từ array)
        if isinstance(img_source, str):
            img = Image.open(img_source)
        else:
            img = Image.fromarray(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB))
        
        base_height = 800
        h_percent = (base_height / float(img.size[1]))
        w_size = int((float(img.size[0]) * float(h_percent)))
        img = img.resize((w_size, base_height), Image.Resampling.LANCZOS)
        
        self.tk_image_display = ImageTk.PhotoImage(img)
        self.canvas_label.config(image=self.tk_image_display, text="")

    # --- KẾT NỐI VỚI LOGIC ---
    def on_click_run(self):
        # 1. Validate dữ liệu
        # Chỉ bắt buộc chọn ảnh gốc. Tiles folder có thể để trống (để dùng mặc định/fallback)
        if not self.target_image_path:
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn ảnh gốc!")
            return
        
        try:
            tile_size = int(self.entry_tile_size.get())
            blend = self.slider_blend.get()
            
            # Lấy giá trị scale (kí tự đầu tiên)
            scale_str = self.combo_scale.get()
            scale_factor = int(scale_str[0]) 
            
            use_adaptive = self.var_adaptive.get()

        except:
            messagebox.showerror("Lỗi", "Thông số không hợp lệ")
            return

        # 2. Khóa UI
        self.btn_run.config(state=tk.DISABLED, text="Đang xử lý...")
        self.btn_save.config(state=tk.DISABLED) # Disable nút lưu khi đang chạy mới
        
        # 3. Khởi tạo Logic Object
        # Nếu chưa chọn folder, truyền None/Empty string để Core tự xử lý fallback
        folder_to_use = self.tiles_folder_path if self.tiles_folder_path else ""

        self.processor = MosaicGenerator(
            target_path=self.target_image_path,
            tiles_folder=folder_to_use,
            tile_size=tile_size,
            blend_factor=blend,
            scale_factor=scale_factor,
            use_adaptive=use_adaptive
        )

        threading.Thread(target=self.run_process_thread, daemon=True).start()

    def run_process_thread(self):
        try:
            # Gọi hàm RUN của Logic và truyền callback vào
            out_path, result_img = self.processor.run(self.update_progress_safe)
            
            # Lưu đường dẫn kết quả tạm thời
            self.current_result_path = out_path

            # Xử lý khi xong (Dùng lambda vẫn an toàn với các biến local bình thường)
            self.root.after(0, lambda: self.show_image_preview(result_img))
            self.root.after(0, lambda: messagebox.showinfo("Thành công", f"Đã tạo xong! Hãy bấm nút 'Lưu' để tải về."))
            self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL)) # Enable nút lưu
            
        except Exception as e:

            error_message = str(e) 
            self.root.after(0, lambda: messagebox.showerror("Lỗi", error_message))
            
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="TẠO ẢNH (Logic Separate)"))

    def update_progress_safe(self, percent, message):
        self.root.after(0, lambda: self._update_ui_elements(percent, message))

    def _update_ui_elements(self, percent, message):
        self.progress['value'] = percent
        self.lbl_status.config(text=message)