import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import cv2

from algorithms.mosaic_core import MosaicGenerator

class PhotomosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Mosaic - MVC Separation Architecture")
        self.root.geometry("1200x850")
        
        self.target_image_path = None
        self.tiles_folder_path = None
        self.tk_image_display = None 
        
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

        # Nút Chạy
        self.btn_run = tk.Button(control_frame, text="TẠO ẢNH (Logic Separate)", command=self.on_click_run, 
                                 bg="#28a745", fg="white", font=("Segoe UI", 12, "bold"), height=2)
        self.btn_run.pack(fill=tk.X, pady=30)
        
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
        if not self.target_image_path or not self.tiles_folder_path:
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn đủ ảnh và folder!")
            return
        
        try:
            tile_size = int(self.entry_tile_size.get())
            blend = self.slider_blend.get()
        except:
            messagebox.showerror("Lỗi", "Thông số không hợp lệ")
            return

        # 2. Khóa UI
        self.btn_run.config(state=tk.DISABLED, text="Đang xử lý...")
        
        # 3. Khởi tạo Logic Object
        self.processor = MosaicGenerator(
            target_path=self.target_image_path,
            tiles_folder=self.tiles_folder_path,
            tile_size=tile_size,
            blend_factor=blend
        )

        threading.Thread(target=self.run_process_thread, daemon=True).start()

    def run_process_thread(self):
        try:
            # Gọi hàm RUN của Logic và truyền callback vào
            out_path, result_img = self.processor.run(self.update_progress_safe)
            
            # Xử lý khi xong (Dùng lambda vẫn an toàn với các biến local bình thường)
            self.root.after(0, lambda: self.show_image_preview(result_img))
            self.root.after(0, lambda: messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {out_path}"))
            
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