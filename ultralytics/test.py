# import cv2
# import numpy as np

# # 检查你的深度图
# depth_path = "./000000000009.png"
# depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# print(f"Shape: {depth.shape}")
# print(f"Dtype: {depth.dtype}")
# print(f"Min: {depth.min()}, Max: {depth.max()}")

# if depth.ndim == 3:
#     print("⚠️ 彩色深度图!")
#     # 检查是否是伪彩色可视化
#     gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
#     print(f"转换后范围: {gray.min()} - {gray.max()}")
    
#     # 检查是否所有通道相同(真灰度图误存为 3 通道)
#     if np.allclose(depth[:,:,0], depth[:,:,1]) and np.allclose(depth[:,:,1], depth[:,:,2]):
#         print("✅ 实际是灰度图,但保存为 3 通道")
#     else:
#         print("❌ 真正的彩色图,可能是可视化图,不适合训练!")
# else:
#     print("✅ 灰度深度图")

# Check which epoch has the best fitness score
import csv

csv_path = './runs/depth/train13/results.csv'

# Read CSV file
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Calculate fitness for each epoch
best_fitness = float('-inf')
best_epoch = -1

print(f"Total epochs: {len(rows)}\n")
print(f"{'Epoch':<8} {'MAE':<10} {'RMSE':<10} {'delta1':<10} {'Fitness':<10}")
print("-" * 58)

for i, row in enumerate(rows):
    epoch = int(row['epoch'])
    mae = float(row['mae'])
    rmse = float(row['rmse'])
    delta1 = float(row['delta1'])
    
    # Calculate fitness: delta1 * 0.7 - mae * 0.2 - rmse * 0.1
    fitness = delta1 * 0.7 - mae * 0.2 - rmse * 0.1
    
    # Print every 10 epochs + last 5 epochs
    if i % 10 == 0 or i >= len(rows) - 5:
        print(f"{epoch:<8} {mae:<10.4f} {rmse:<10.4f} {delta1:<10.4f} {fitness:<10.4f}")
    
    if fitness > best_fitness:
        best_fitness = fitness
        best_epoch = epoch

print("-" * 58)
print(f"\n✅ 最优 epoch: {best_epoch}")
print(f"✅ 最高 fitness: {best_fitness:.4f}")

# Show best epoch details
best_row = rows[best_epoch - 1]
print(f"\n最优 epoch 的详细指标:")
print(f"  MAE:    {float(best_row['mae']):.4f}")
print(f"  RMSE:   {float(best_row['rmse']):.4f}")
print(f"  AbsRel: {float(best_row['abs_rel']):.4f}")
print(f"  delta1: {float(best_row['delta1']):.4f}")
print(f"  delta2: {float(best_row['delta2']):.4f}")
print(f"  delta3: {float(best_row['delta3']):.4f}")
