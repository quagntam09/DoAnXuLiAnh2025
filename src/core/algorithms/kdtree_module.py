import numpy as np


class _Node:
    # __slots__ helps reduce memory usage for many nodes.
    __slots__ = ("location_idx", "axis", "left", "right", "is_leaf", "indices")

    def __init__(self, location_idx=None, axis=None, left=None, right=None, indices=None):
        self.location_idx = location_idx
        self.axis = axis
        self.left = left
        self.right = right
        self.indices = indices
        self.is_leaf = indices is not None


class KDTree:
    def __init__(self, points: np.ndarray, leaf_size: int = 16):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points must be a 2D array with shape (N, k)")

        self.points = pts
        self.n, self.k = pts.shape
        self.leaf_size = leaf_size

        idxs = np.arange(self.n, dtype=np.int32)
        self.root = self._build(idxs, depth=0)

    def _build(self, idxs: np.ndarray, depth: int):
        n_points = idxs.size
        if n_points == 0:
            return None

        if n_points <= self.leaf_size:
            return _Node(indices=idxs)

        axis = depth % self.k
        mid = n_points // 2
        vals = self.points[idxs, axis]
        partition_idx = np.argpartition(vals, mid)
        idxs = idxs[partition_idx]
        node_idx = idxs[mid]

        return _Node(
            location_idx=node_idx,
            axis=axis,
            left=self._build(idxs[:mid], depth + 1),
            right=self._build(idxs[mid + 1 :], depth + 1),
        )

    def query(self, point):
        """Return (distance, index) of nearest point."""
        target = np.asarray(point, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.k:
            raise ValueError(f"point must have shape ({self.k},)")
        points = self.points

        best_d2 = float("inf")
        best_idx = -1

        def _search(node):
            nonlocal best_d2, best_idx
            if node is None:
                return

            if node.is_leaf:
                leaf_pts = points[node.indices]
                diff = leaf_pts - target
                d2_arr = np.einsum("ij,ij->i", diff, diff)
                min_idx_in_leaf = np.argmin(d2_arr)
                min_d2 = d2_arr[min_idx_in_leaf]

                if min_d2 < best_d2:
                    best_d2 = min_d2
                    best_idx = node.indices[min_idx_in_leaf]
                return

            p = points[node.location_idx]
            diff_p = p - target
            d2 = float(np.dot(diff_p, diff_p))

            if d2 < best_d2:
                best_d2 = d2
                best_idx = node.location_idx

            diff_axis = target[node.axis] - p[node.axis]
            near = node.left if diff_axis < 0 else node.right
            far = node.right if diff_axis < 0 else node.left

            _search(near)
            if diff_axis * diff_axis < best_d2:
                _search(far)

        _search(self.root)
        return float(np.sqrt(best_d2)), int(best_idx)


__all__ = ["KDTree"]
