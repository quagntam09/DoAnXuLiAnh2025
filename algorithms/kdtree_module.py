import math

class Node:
    # Thêm original_index để lưu vị trí gốc của ảnh trong danh sách
    def __init__(self, point, original_index, left=None, right=None, axis=0):
        self.point = point
        self.original_index = original_index 
        self.left = left    
        self.right = right 
        self.axis = axis    

class KDTree:
    def __init__(self, points):
        """
        points: Là một list các vector màu. Ví dụ: [[255, 0, 0], [0, 255, 0]...]
        """
        # Gắn index vào từng điểm trước khi xây cây: [(point, index), ...]
        # Để sau này khi sort không bị mất dấu vị trí ban đầu
        indexed_points = []
        for i in range(len(points)):
            indexed_points.append((points[i], i))

        if len(points) == 0:
            raise ValueError("Dữ liệu đầu vào không được rỗng")
        
        self.k = len(points[0]) 
        self.root = self._build_tree(indexed_points, depth=0)

    def _build_tree(self, points_with_index, depth):
        # Dùng len() để kiểm tra rỗng (An toàn tuyệt đối với mọi kiểu dữ liệu)
        if len(points_with_index) == 0:
            return None

        axis = depth % self.k

        # Sort dựa trên giá trị trục (x[0] là point màu, x[1] là index)
        points_with_index.sort(key=lambda x: x[0][axis])

        median_idx = len(points_with_index) // 2
        median_val = points_with_index[median_idx] # (point, index)

        return Node(
            point=median_val[0],
            original_index=median_val[1], # Lưu index vào nút
            left=self._build_tree(points_with_index[:median_idx], depth + 1),
            right=self._build_tree(points_with_index[median_idx + 1:], depth + 1),
            axis=axis
        )

    def _distance_squared(self, point1, point2):
        dist = 0
        for i in range(self.k):
            dist += (point1[i] - point2[i]) ** 2
        return dist

    def find_nearest(self, target_point):

        # Tìm điểm gần nhất.
        # Trả về: (original_index, distance)

        best_node = [None] 
        best_dist_sq = [float('inf')] 
        best_index = [-1] # Biến để lưu index của ảnh tìm được

        def _search_recursive(node):
            if node is None:
                return

            dist_sq = self._distance_squared(target_point, node.point)

            # Nếu tìm thấy điểm tốt hơn -> Cập nhật cả khoảng cách lẫn Index
            if dist_sq < best_dist_sq[0]:
                best_dist_sq[0] = dist_sq
                best_node[0] = node.point
                best_index[0] = node.original_index 

            axis = node.axis
            diff = target_point[axis] - node.point[axis]
            
            near_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _search_recursive(near_branch)

            # Cắt tỉa (Pruning)
            if diff**2 < best_dist_sq[0]:
                _search_recursive(far_branch)

        _search_recursive(self.root)
        
        # Trả về Index (để lấy ảnh) và Distance
        return best_index[0], math.sqrt(best_dist_sq[0])