import cv2
import numpy as np
import pickle

def find_valid_square(pt, depth, size=10, step=2, min_ratio=0.5, max_iter=20):
    """在depth图上以pt为中心找一个有效深度占比>=min_ratio的正方形区域，否则朝有效率最大方向移动"""
    h, w = depth.shape
    x, y = pt
    for _ in range(max_iter):
        x1 = max(0, x - size//2)
        y1 = max(0, y - size//2)
        x2 = min(w, x + size//2 + 1)
        y2 = min(h, y + size//2 + 1)
        roi = depth[y1:y2, x1:x2]
        valid = np.count_nonzero(roi)
        total = roi.size
        ratio = valid / total if total > 0 else 0
        if ratio >= min_ratio:
            return (x, y), roi
        # 计算四个方向移动后的有效率
        dirs = [(-step,0),(step,0),(0,-step),(0,step)]
        best_ratio = ratio
        best_dir = (0,0)
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            nx1 = max(0, nx - size//2)
            ny1 = max(0, ny - size//2)
            nx2 = min(w, nx + size//2 + 1)
            ny2 = min(h, ny + size//2 + 1)
            nroi = depth[ny1:ny2, nx1:nx2]
            nvalid = np.count_nonzero(nroi)
            ntotal = nroi.size
            nratio = nvalid / ntotal if ntotal > 0 else 0
            if nratio > best_ratio:
                best_ratio = nratio
                best_dir = (dx, dy)
        if best_dir == (0,0):
            break  # 四个方向都不如当前，不再移动
        x += best_dir[0]
        y += best_dir[1]
    return (x, y), depth[max(0, y-size//2):min(h, y+size//2+1), max(0, x-size//2):min(w, x+size//2+1)]

# 读取深度图并伪彩色化
mask = cv2.imread('/home/xuan/dianrobot/wjx/eye/get_r/imgs/imgsdepth_mask.png', cv2.IMREAD_UNCHANGED)
depth = cv2.imread('/home/xuan/dianrobot/wjx/eye/get_r/imgs/test_camera_2_depth.png', cv2.IMREAD_UNCHANGED)
depth = cv2.bitwise_and(depth, depth, mask=mask)
min_depth = 600
max_depth = 700
depth_clip = np.clip(depth, min_depth, max_depth)
depth_norm = ((depth_clip - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

with open('/home/xuan/dianrobot/wjx/eye/get_r/imgs/hand_points_pixel.pkl', 'rb') as f:
    points = pickle.load(f)
points = points[:9]

size = 20 # 正方形边长
step = 5   # 步长
for pt in points:
    (x, y), roi = find_valid_square(pt, depth, size=size, step=step, min_ratio=0.8)
    # 画正方形
    x1 = int(x - size//2)
    y1 = int(y - size//2)
    x2 = int(x + size//2)
    y2 = int(y + size//2)
    cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0,255,255), 2)
    # 画中心点
    cv2.circle(depth_color, (int(x), int(y)), 2, (0,0,255), -1)

cv2.imshow('Depth with Squares', depth_color)
cv2.imwrite('/home/xuan/dianrobot/wjx/eye/get_r/imgs/depth_with_squares.png', depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()