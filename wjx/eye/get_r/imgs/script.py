import cv2
import numpy as np

# 读取深度图（假设为16位单通道PNG）
depth = cv2.imread('/home/xuan/dianrobot/wjx/eye/get_photos/data2/depth/depth001_2.png', cv2.IMREAD_UNCHANGED)
if depth is None:
    raise FileNotFoundError("未找到 test_depth.png 文件")

# 乘以100，注意防止溢出，转换为更高位类型
depth_new = (depth.astype(np.uint16) * 255).astype(np.uint16)

# 保存为16位或32位PNG（推荐32位以防溢出）
cv2.imwrite('wjx/eye/get_r/imgs/depth_x100.png', depth_new)
print("已保存 depth_x100.png")