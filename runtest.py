from algorithms.kdtree_module import KDTree

points = [
    [2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]
]
tree = KDTree(points)

target = [9, 2]
result, distance = tree.find_nearest(target)

print(f"Điểm cần tìm: {target}")
print(f"Điểm gần nhất trong dữ liệu: {result}")
print(f"Khoảng cách: {distance:.2f}")