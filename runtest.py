from algorithms.kdtree_module import KDTree

points = [
    [2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2], [10, 2], [9, 1]
]
# Indices:
# 0: [2, 3]
# 1: [5, 4]
# 2: [9, 6]
# 3: [4, 7]
# 4: [8, 1]
# 5: [7, 2]
# 6: [10, 2]
# 7: [9, 1]

tree = KDTree(points)

target = [9, 2]
print(f"Điểm cần tìm: {target}")

# Test 1 nearest
idx, dist = tree.find_nearest(target)
print(f"1 điểm gần nhất: Index={idx}, Point={points[idx]}, Dist={dist:.2f}")

# Test k nearest
k = 3
results = tree.find_k_nearest(target, k)
print(f"\n{k} điểm gần nhất:")
for idx, dist in results:
    print(f"Index={idx}, Point={points[idx]}, Dist={dist:.2f}")

# Expected:
# [9, 2] matches [9, 1] (dist 1.0) or [8, 1] (dist sqrt(2)=1.41) or [10, 2] (dist 1.0)
# Let's see what it outputs.
