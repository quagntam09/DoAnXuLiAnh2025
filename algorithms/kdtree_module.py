import math

class Node:
    def __init__(self, point, left=None, right=None, axis=0):
        self.point = point  
        self.left = left    
        self.right = right 
        self.axis = axis    

class KDTree:
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Dữ liệu đầu vào không được rỗng")
        
        self.k = len(points[0]) 
        self.root = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        if not points:
            return None

        axis = depth % self.k

        points.sort(key=lambda x: x[axis])

   
        median_idx = len(points) // 2
        median_point = points[median_idx]

        return Node(
            point=median_point,
            left=self._build_tree(points[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], depth + 1),
            axis=axis
        )

    def _distance_squared(self, point1, point2):
        dist = 0
        for i in range(self.k):
            dist += (point1[i] - point2[i]) ** 2
        return dist

    def find_nearest(self, target_point):
        best_node = [None] 
        best_dist_sq = [float('inf')] 

        def _search_recursive(node):
            if node is None:
                return

            dist_sq = self._distance_squared(target_point, node.point)

            if dist_sq < best_dist_sq[0]:
                best_dist_sq[0] = dist_sq
                best_node[0] = node.point

            axis = node.axis
            diff = target_point[axis] - node.point[axis]
            
            near_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _search_recursive(near_branch)

            if diff**2 < best_dist_sq[0]:
                _search_recursive(far_branch)

        _search_recursive(self.root)
        
        return best_node[0], math.sqrt(best_dist_sq[0])