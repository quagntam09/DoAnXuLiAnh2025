import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.common.config import (
    DEFAULT_BLEND,
    DEFAULT_FRAME_EVERY,
    DEFAULT_LEVELS,
    DEFAULT_TILE_SIZE,
)
from src.common.exceptions import InvalidInputError, MosaicError
from src.core.services.mosaic_service import MosaicService
from src.infra.image_io import read_image, write_image


def bgr_to_tk(img_bgr: np.ndarray, max_w=800, max_h=800) -> ImageTk.PhotoImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
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

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#333")

        self.target_path = tk.StringVar(value="")
        self.tiles_folder = tk.StringVar(value="")
        self.tile_size = tk.IntVar(value=DEFAULT_TILE_SIZE)
        self.levels = tk.IntVar(value=DEFAULT_LEVELS)
        self.blend = tk.DoubleVar(value=DEFAULT_BLEND)

        self._result_img = None
        self._photo = None
        self._build_ui()

    def _build_ui(self):
        main_paned = tk.PanedWindow(self, orient="horizontal", sashwidth=5, bg="#dcdcdc")
        main_paned.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_paned, padding=15)
        main_paned.add(left_frame, minsize=350, width=380)
        lbl_title = ttk.Label(left_frame, text="🛠 BẢNG ĐIỀU KHIỂN", style="Header.TLabel")
        lbl_title.pack(anchor="w", pady=(0, 15))

        grp_input = ttk.LabelFrame(left_frame, text="1. Chọn Dữ Liệu", padding=10)
        grp_input.pack(fill="x", pady=5)
        ttk.Label(grp_input, text="Ảnh gốc (Chủ đề):").pack(anchor="w")
        ttk.Button(grp_input, text="📂 Mở ảnh gốc...", command=self.pick_target).pack(fill="x", pady=(2, 8))
        self.lbl_target_name = ttk.Label(grp_input, text="(Chưa chọn ảnh)", foreground="gray", wraplength=300)
        self.lbl_target_name.pack(anchor="w", pady=(0, 10))
        ttk.Label(grp_input, text="Kho ảnh ghép (Dataset):").pack(anchor="w")
        ttk.Button(grp_input, text="📂 Chọn thư mục ảnh nhỏ...", command=self.pick_tiles_folder).pack(fill="x", pady=(2, 8))
        self.lbl_tiles_name = ttk.Label(grp_input, text="(Chưa chọn thư mục)", foreground="gray", wraplength=300)
        self.lbl_tiles_name.pack(anchor="w")

        grp_config = ttk.LabelFrame(left_frame, text="2. Tùy Chỉnh Nghệ Thuật", padding=10)
        grp_config.pack(fill="x", pady=15)
        self.lbl_tile_val = ttk.Label(grp_config, text=f"Kích thước ô nhỏ: {self.tile_size.get()} px")
        self.lbl_tile_val.pack(anchor="w")
        ttk.Scale(
            grp_config,
            from_=5,
            to=80,
            variable=self.tile_size,
            command=lambda v: self.lbl_tile_val.config(text=f"Kích thước ô nhỏ: {int(float(v))} px"),
        ).pack(fill="x", pady=(0, 10))

        self.lbl_level_val = ttk.Label(grp_config, text=f"Độ phân giải (Levels): {self.levels.get()}")
        self.lbl_level_val.pack(anchor="w")
        ttk.Scale(
            grp_config,
            from_=1,
            to=6,
            variable=self.levels,
            command=lambda v: self.lbl_level_val.config(text=f"Độ phân giải (Levels): {int(float(v))}"),
        ).pack(fill="x", pady=(0, 10))

        self.lbl_blend_val = ttk.Label(grp_config, text=f"Pha trộn ảnh gốc: {int(self.blend.get() * 100)}%")
        self.lbl_blend_val.pack(anchor="w")
        ttk.Scale(
            grp_config,
            from_=0.0,
            to=1.0,
            variable=self.blend,
            command=lambda v: self.lbl_blend_val.config(text=f"Pha trộn ảnh gốc: {int(float(v) * 100)}%"),
        ).pack(fill="x")
        ttk.Label(
            grp_config,
            text="(Kéo cao để ảnh rõ nét hơn, thấp để nghệ thuật hơn)",
            font=("Arial", 8, "italic"),
            foreground="gray",
        ).pack(anchor="w")

        grp_action = ttk.LabelFrame(left_frame, text="3. Thực Hiện", padding=10)
        grp_action.pack(fill="x", pady=5)
        self.btn_run = ttk.Button(grp_action, text="▶ BẮT ĐẦU TẠO TRANH", command=self.run_mosaic)
        self.btn_run.pack(fill="x", pady=5)
        self.progress = ttk.Progressbar(grp_action, mode="determinate")
        self.progress.pack(fill="x", pady=5)
        self.status = tk.StringVar(value="Sẵn sàng.")
        self.lbl_status = ttk.Label(grp_action, textvariable=self.status, foreground="blue", wraplength=300)
        self.lbl_status.pack(fill="x")
        self.btn_save = ttk.Button(left_frame, text="💾 Lưu Kết Quả Về Máy", command=self.save_as, state="disabled")
        self.btn_save.pack(fill="x", pady=20, side="bottom")

        right_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(right_frame)
        self.preview_container = tk.Label(right_frame, bg="#333333", text="Khu vực hiển thị ảnh", fg="white")
        self.preview_container.pack(fill="both", expand=True)

    def pick_target(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh gốc",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return

        self.target_path.set(path)
        self.lbl_target_name.config(text=f"✔ {os.path.basename(path)}", foreground="green")
        try:
            img = read_image(path)
        except MosaicError as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        self._result_img = None
        self.btn_save.config(state="disabled")
        self.show_image(img)
        self.status.set("Đã tải ảnh gốc.")

    def pick_tiles_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa tập ảnh nhỏ")
        if folder:
            self.tiles_folder.set(folder)
            self.lbl_tiles_name.config(text=f"✔ .../{os.path.basename(folder)}", foreground="green")
            self.status.set("Đã chọn kho ảnh mẫu.")

    def show_image(self, img_bgr: np.ndarray):
        w = self.preview_container.winfo_width()
        h = self.preview_container.winfo_height()
        if w < 100:
            w = 800
        if h < 100:
            h = 600
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

        self.btn_run.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.progress["value"] = 0
        self.status.set("Đang khởi động thuật toán...")

        t_size = int(self.tile_size.get())
        levs = int(self.levels.get())
        bl = float(self.blend.get())

        def on_progress(p, msg):
            self.after(0, lambda: self.progress.configure(value=float(p)))
            self.after(0, lambda: self.status.set(msg))

        def on_frame(frame_img):
            show_img = frame_img.copy()
            self.after(0, lambda: self.show_image(show_img))

        def worker_thread():
            try:
                service = MosaicService(
                    target_path=target,
                    tiles_folder=tiles,
                    tile_size=t_size,
                    blend_factor=bl,
                    levels=levs,
                    frame_every=DEFAULT_FRAME_EVERY,
                )
                final_img = service.run(progress_callback=on_progress, frame_callback=on_frame)
                self._result_img = final_img
                self.after(0, lambda: self.show_image(final_img))
                self.after(0, lambda: self.btn_save.config(state="normal"))
                self.after(0, lambda: messagebox.showinfo("Hoàn tất", "Đã tạo tranh Mosaic thành công!"))
            except InvalidInputError as exc:
                self.after(0, lambda err=str(exc): messagebox.showwarning("Dữ liệu không hợp lệ", err))
            except MosaicError as exc:
                self.after(0, lambda err=str(exc): messagebox.showerror("Lỗi Runtime", err))
            except Exception as exc:
                self.after(0, lambda err=str(exc): messagebox.showerror("Lỗi Runtime", f"Có lỗi xảy ra:\n{err}"))
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
            filetypes=[("JPG Image", "*.jpg"), ("PNG Image", "*.png")],
        )
        if not path:
            return
        try:
            write_image(path, self._result_img)
            messagebox.showinfo("Đã lưu", f"Ảnh đã được lưu tại:\n{path}")
        except MosaicError as exc:
            messagebox.showerror("Lỗi", str(exc))


if __name__ == "__main__":
    App().mainloop()
