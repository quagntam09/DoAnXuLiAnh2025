import tkinter as tk
import ctypes
from ui.main_window import PhotomosaicApp

if __name__ == "__main__":
    # Kích hoạt High DPI để giao diện sắc nét trên Windows
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = PhotomosaicApp(root)
    root.mainloop()