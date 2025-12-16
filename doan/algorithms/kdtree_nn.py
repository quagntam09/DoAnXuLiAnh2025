import numpy as np
from .kdtree_module import KDTree

class KDTreeNearestNeighbor:

    def __init__(self, colors_arr: np.ndarray):
        if colors_arr is None:
            raise ValueError("colors_arr is None")

        colors_arr = np.asarray(colors_arr, dtype=np.float32)
        if colors_arr.ndim != 2 or colors_arr.shape[1] != 3 or colors_arr.shape[0] == 0:
            raise ValueError("colors_arr phải có shape (N, 3) và N > 0")

        self.colors = colors_arr
        self.tree = KDTree(self.colors)

    def query(self, color: np.ndarray) -> int:
        color = np.asarray(color, dtype=np.float32).reshape(-1)
        if color.shape[0] != 3:
            raise ValueError("color phải có shape (3,)")

        _, idx = self.tree.query(color)
        return int(idx)
