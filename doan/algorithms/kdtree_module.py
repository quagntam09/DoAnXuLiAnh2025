import numpy as np

class _Node:
    __slots__ = ("idx", "axis", "left", "right")

    def __init__(self, idx: int, axis: int, left=None, right=None):
        self.idx = idx
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points phải là mảng 2D (N, k) với N > 0")

        self.points = pts
        self.n, self.k = pts.shape

        idxs = np.arange(self.n, dtype=np.int32)
        self.root = self._build(idxs, depth=0)

    def _build(self, idxs: np.ndarray, depth: int):
        if idxs.size == 0:
            return None

        axis = depth % self.k

        # sort idxs theo giá trị points[:, axis] và chọn median
        vals = self.points[idxs, axis]
        order = np.argsort(vals, kind="mergesort")  # stable
        idxs = idxs[order]

        mid = idxs.size // 2
        node_idx = int(idxs[mid])

        left = self._build(idxs[:mid], depth + 1)
        right = self._build(idxs[mid + 1:], depth + 1)

        return _Node(node_idx, axis, left, right)

    @staticmethod
    def _sqdist(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def query(self, point):
        target = np.asarray(point, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.k:
            raise ValueError(f"point phải có đúng {self.k} chiều")

        best_idx = -1
        best_d2 = float("inf")

        def _search(node):
            nonlocal best_idx, best_d2
            if node is None:
                return

            p = self.points[node.idx]
            d2 = self._sqdist(target, p)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = node.idx

            axis = node.axis
            diff = target[axis] - p[axis]

            # nhánh gần và nhánh xa
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left

            _search(near)

            # nếu mặt phẳng phân tách có thể chứa điểm tốt hơn thì kiểm tra nhánh xa
            if diff * diff < best_d2:
                _search(far)

        _search(self.root)

        return float(np.sqrt(best_d2)), int(best_idx)
