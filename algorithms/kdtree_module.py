import numpy as np

class _Node:
    __slots__ = ("idx", "axis", "left", "right")

    def __init__(self, idx: int, axis: int, left=None, right=None):
        self.idx = idx      # Index của điểm trong mảng gốc
        self.axis = axis    # Trục chia cắt (0=x, 1=y, 2=z...)
        self.left = left
        self.right = right

class KDTree:
    """
    Cài đặt KD-Tree đơn giản để tìm Nearest Neighbor.
    """
    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points phải là mảng 2D (N, k) với N > 0")

        self.points = pts
        self.n, self.k = pts.shape

        # Xây dựng cây từ danh sách index
        idxs = np.arange(self.n, dtype=np.int32)
        self.root = self._build(idxs, depth=0)

    def _build(self, idxs: np.ndarray, depth: int):
        if idxs.size == 0:
            return None

        axis = depth % self.k

        # Sắp xếp và chọn median để cây cân bằng
        vals = self.points[idxs, axis]
        order = np.argsort(vals, kind="mergesort")
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
        """Trả về (distance, index) của điểm gần nhất."""
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
            
            # Cập nhật điểm tốt nhất nếu tìm thấy gần hơn
            if d2 < best_d2:
                best_d2 = d2
                best_idx = node.idx

            axis = node.axis
            diff = target[axis] - p[axis]

            # Quy tắc duyệt: ưu tiên nhánh "gần" hơn với target
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left

            _search(near)

            # Pruning: Chỉ duyệt nhánh "xa" nếu khoảng cách tới mặt phẳng cắt < best_dist
            if diff * diff < best_d2:
                _search(far)

        _search(self.root)
        return float(np.sqrt(best_d2)), int(best_idx)