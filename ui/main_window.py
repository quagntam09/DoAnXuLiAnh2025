import tkinter as tk
from tkinter import messagebox
import random
import math

from algorithms.kdtree_module import KDTree

class KDTreeVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("KD-Tree Nearest Neighbor Visualizer")
        
        self.width = 800
        self.height = 600
        
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_gen = tk.Button(control_frame, text="1. Sinh dữ liệu ngẫu nhiên", command=self.generate_data, bg="#dddddd")
        self.btn_gen.pack(side=tk.LEFT, padx=10)

        tk.Label(control_frame, text="2. Click chuột vào vùng trắng để tìm điểm gần nhất", fg="blue").pack(side=tk.LEFT, padx=20)
        
        self.lbl_info = tk.Label(control_frame, text="Distance: ...", font=("Arial", 10, "bold"))
        self.lbl_info.pack(side=tk.RIGHT, padx=20)

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white", cursor="cross")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.points = []
        self.tree = None
        self.target_id = None 
        self.line_id = None   

    def generate_data(self):
        self.points = []
        self.canvas.delete("all") 
        
        for _ in range(50):
            x = random.randint(20, self.width - 20)
            y = random.randint(20, self.height - 20)
            self.points.append([x, y])
            
            r = 3
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

        try:
            self.tree = KDTree(self.points)
            messagebox.showinfo("Thành công", "Đã xây dựng KD-Tree với 50 điểm ngẫu nhiên!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xây dựng cây: {e}")

    def on_canvas_click(self, event):
        if not self.tree:
            messagebox.showwarning("Chưa có dữ liệu", "Vui lòng nhấn nút 'Sinh dữ liệu ngẫu nhiên' trước!")
            return

        target = [event.x, event.y]

        nearest_point, distance = self.tree.find_nearest(target)

        self.draw_result(target, nearest_point, distance)

    def draw_result(self, target, nearest, distance):
        self.canvas.delete("result_layer")

        tx, ty = target
        r = 5
        self.canvas.create_oval(tx-r, ty-r, tx+r, ty+r, fill="red", outline="red", tags="result_layer")
        
        nx, ny = nearest
        self.canvas.create_line(tx, ty, nx, ny, fill="blue", dash=(4, 4), width=2, tags="result_layer")

        self.canvas.create_oval(nx-6, ny-6, nx+6, ny+6, outline="green", width=3, tags="result_layer")

        self.lbl_info.config(text=f"Khoảng cách: {distance:.2f} px")

def create_app():
    root = tk.Tk()
    app = KDTreeVisualizer(root)
    root.mainloop()