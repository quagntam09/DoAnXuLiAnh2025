import os
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from mosaic_core import MosaicGenerator


def bgr_to_tk(img_bgr: np.ndarray, max_w=700, max_h=320) -> ImageTk.PhotoImage:
    """Convert BGR numpy -> Tk PhotoImage (resize fit)."""
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
        self.title("Photomosaic Generator")
        self.geometry("900x620")

        self.target_path = tk.StringVar(value="")
        self.tiles_folder = tk.StringVar(value="")
        self.tile_size = tk.IntVar(value=10)
        self.levels = tk.IntVar(value=3)
        self.blend = tk.DoubleVar(value=0.2)

        self._current_img = None
        self._result_img = None
        self._photo = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=12)

        ttk.Label(top, text="Ảnh gốc (target):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.target_path, width=70).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(top, text="Chọn...", command=self.pick_target).grid(row=0, column=2, **pad)

        ttk.Label(top, text="Dataset tiles (folder):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.tiles_folder, width=70).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(top, text="Chọn...", command=self.pick_tiles_folder).grid(row=1, column=2, **pad)

        params = ttk.LabelFrame(top, text="Tham số")
        params.grid(row=2, column=0, columnspan=3, sticky="we", **pad)

        ttk.Label(params, text="Kích thước ảnh ghép (tile):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(params, from_=4, to=64, textvariable=self.tile_size, width=8).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(params, text="Số level ghép ảnh (thô → mịn):").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(params, from_=1, to=5, textvariable=self.levels, width=8).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(params, text="Độ pha trộn với ảnh gốc:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.05, textvariable=self.blend, width=8).grid(row=0, column=5, sticky="w", **pad)

        top.columnconfigure(1, weight=1)

        mid = ttk.LabelFrame(self, text="Preview")
        mid.pack(fill="both", expand=True, padx=12, pady=6)

        self.canvas = tk.Label(mid, text="Chưa có ảnh", anchor="center")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=12, pady=12)

        self.progress = ttk.Progressbar(bot, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=6, pady=6)

        self.status = tk.StringVar(value="Sẵn sàng.")
        ttk.Label(bot, textvariable=self.status).pack(anchor="w", padx=6)

        btns = ttk.Frame(bot)
        btns.pack(fill="x", padx=6, pady=6)

        self.btn_run = ttk.Button(btns, text="Chạy Mosaic", command=self.run_mosaic)
        self.btn_run.pack(side="left", padx=6)

        self.btn_save = ttk.Button(btns, text="Save As...", command=self.save_as, state="disabled")
        self.btn_save.pack(side="left", padx=6)

    def pick_target(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh gốc",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self.target_path.set(path)

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh gốc.")
            return

        self._current_img = img
        self._result_img = None
        self.btn_save.config(state="disabled")
        self.show_image(img)
        self.status.set("Đã load ảnh gốc.")

        self.update_idletasks()
        self.minsize(self.winfo_reqwidth(), self.winfo_reqheight())

    def pick_tiles_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục dataset tiles")
        if folder:
            self.tiles_folder.set(folder)

    def show_image(self, img_bgr: np.ndarray):
        self._photo = bgr_to_tk(img_bgr)
        self.canvas.configure(image=self._photo, text="")

    def run_mosaic(self):
        target = self.target_path.get().strip()
        tiles = self.tiles_folder.get().strip()

        if not target or not os.path.isfile(target):
            messagebox.showerror("Lỗi", "Bạn chưa chọn ảnh gốc hợp lệ.")
            return
        if not tiles or not os.path.isdir(tiles):
            messagebox.showerror("Lỗi", "Bạn chưa chọn dataset tiles hợp lệ.")
            return

        self.btn_run.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.progress["value"] = 0
        self.status.set("Đang chạy...")

        def progress_cb(p, msg):
            def _u():
                self.progress["value"] = float(p)
                if msg:
                    self.status.set(msg)
            self.after(0, _u)

        def frame_cb(mosaic_bgr):
            self.after(0, lambda: self.show_image(mosaic_bgr))


        def worker():
            try:
                gen = MosaicGenerator(
                    target_path=target,
                    tiles_folder=tiles,
                    tile_size=int(self.tile_size.get()),
                    blend_factor=float(self.blend.get()),
                    levels=int(self.levels.get())
                )
                img = gen.run(progress_cb, frame_callback=frame_cb)
                self._result_img = img
                self.after(0, lambda im=img: self.show_image(im))
                self.after(0, lambda: self.btn_save.config(state="normal"))
                self.after(0, lambda: messagebox.showinfo("Xong", "Hoàn tất! Bạn có thể bấm Save As... để lưu."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            finally:
                self.after(0, lambda: self.btn_run.config(state="normal"))
                self.after(0, lambda: self.status.set("Sẵn sàng."))

        threading.Thread(target=worker, daemon=True).start()

    def save_as(self):
        if self._result_img is None:
            return
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh mosaic",
            defaultextension=".jpg",
            filetypes=[("JPG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")]
        )
        if not path:
            return
        ok = cv2.imwrite(path, self._result_img)
        if not ok:
            messagebox.showerror("Lỗi", "Lưu file thất bại.")
        else:
            messagebox.showinfo("Đã lưu", f"Đã lưu tại:\n{path}")


if __name__ == "__main__":
    App().mainloop()
