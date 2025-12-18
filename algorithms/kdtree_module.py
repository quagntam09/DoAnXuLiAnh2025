import numpy as np

class _Node:
    # __slots__ giúp tiết kiệm bộ nhớ khi tạo hàng nghìn node
    __slots__ = ("location_idx", "axis", "left", "right", "is_leaf", "indices")

    def __init__(self, location_idx=None, axis=None, left=None, right=None, indices=None):
        self.location_idx = location_idx # Index của điểm chốt (nếu không phải leaf)
        self.axis = axis                 # Trục chia cắt
        self.left = left
        self.right = right
        self.indices = indices           # Danh sách index các điểm (nếu là leaf)
        self.is_leaf = (indices is not None)

class KDTree:
    def __init__(self, points: np.ndarray, leaf_size: int = 16):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points phải là mảng 2D (N, k)")

        self.points = pts
        self.n, self.k = pts.shape
        self.leaf_size = leaf_size # Ngưỡng để dừng chia nhỏ và lưu thành cụm

        idxs = np.arange(self.n, dtype=np.int32)
        self.root = self._build(idxs, depth=0)

    def _build(self, idxs: np.ndarray, depth: int):
        n_points = idxs.size
        if n_points == 0:
            return None

        # Nếu số điểm ít, gộp thành 1 lá để xử lý vector hóa sau này, giảm độ sâu cây
        if n_points <= self.leaf_size:
            return _Node(indices=idxs)

        axis = depth % self.k
        mid = n_points // 2
        
        # Chỉ đảm bảo phần tử ở 'mid' là median, bên trái nhỏ hơn, bên phải lớn hơn.
        # Nhanh hơn argsort rất nhiều (O(N) so với O(N log N))
        vals = self.points[idxs, axis]
        partition_idx = np.argpartition(vals, mid)
        idxs = idxs[partition_idx]
        
        # Điểm median được chọn làm chốt (pivot)
        node_idx = idxs[mid]
        
        # Chia trị
        return _Node(
            location_idx=node_idx,
            axis=axis,
            left=self._build(idxs[:mid], depth + 1),
            right=self._build(idxs[mid + 1:], depth + 1)
        )

    def query(self, point):
        """Trả về (distance, index) của điểm gần nhất."""
        target = np.asarray(point, dtype=np.float32).reshape(-1)
        
        # Biến cục bộ để truy cập nhanh hơn
        points = self.points
        
        best_d2 = float("inf")
        best_idx = -1

        def _search(node):
            nonlocal best_d2, best_idx
            if node is None:
                return

            if node.is_leaf:
                # Lấy tất cả điểm trong leaf
                leaf_pts = points[node.indices]
                
                # Tính khoảng cách Euclidean bình phương tới tất cả điểm cùng lúc
                diff = leaf_pts - target
                # np.einsum thường nhanh hơn np.sum(diff**2, axis=1)
                d2_arr = np.einsum('ij,ij->i', diff, diff)
                
                # Tìm min trong bucket này
                min_idx_in_leaf = np.argmin(d2_arr)
                min_d2 = d2_arr[min_idx_in_leaf]
                
                if min_d2 < best_d2:
                    best_d2 = min_d2
                    best_idx = node.indices[min_idx_in_leaf]
                return

            # Xử lý Node thường (Pivot)
            p = points[node.location_idx]
            
            # Tính khoảng cách tới pivot
            # Unrolled distance cho 3 chiều (nhanh hơn loop/numpy cho 1 điểm)
            d0 = p[0] - target[0]
            d1 = p[1] - target[1]
            d2_ = p[2] - target[2]
            d2 = d0*d0 + d1*d1 + d2_*d2_

            if d2 < best_d2:
                best_d2 = d2
                best_idx = node.location_idx

            # Quy tắc duyệt nhánh
            diff_axis = target[node.axis] - p[node.axis]
            near = node.left if diff_axis < 0 else node.right
            far = node.right if diff_axis < 0 else node.left

            _search(near)

            # Pruning (Cắt tỉa nhánh)
            if diff_axis * diff_axis < best_d2:
                _search(far)

        _search(self.root)
        return float(np.sqrt(best_d2)), int(best_idx)