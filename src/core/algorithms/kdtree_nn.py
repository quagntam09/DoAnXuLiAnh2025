import numpy as np

from src.core.algorithms.kdtree_module import KDTree


class KDTreeNearestNeighbor:
    def __init__(self, colors_arr: np.ndarray):
        if colors_arr is None:
            raise ValueError("colors_arr is None")

        colors_arr = np.asarray(colors_arr, dtype=np.float32)
        if colors_arr.ndim != 2 or colors_arr.shape[0] == 0:
            raise ValueError("colors_arr must have shape (N, d) and N > 0")

        self.colors = colors_arr
        self.dims = int(colors_arr.shape[1])
        self.tree = KDTree(self.colors)

    def query(self, color: np.ndarray) -> int:
        color = np.asarray(color, dtype=np.float32).reshape(-1)
        if color.shape[0] != self.dims:
            raise ValueError(f"color must have shape ({self.dims},)")

        # KDTree.query returns (distance, index)
        _, idx = self.tree.query(color)
        return int(idx)


__all__ = ["KDTreeNearestNeighbor"]