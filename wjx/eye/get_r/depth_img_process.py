'''
处理手掌的深度图像,分割出拇指和食指区域内的有效深度点
'''

import cv2
import numpy as np
import SpaceMath


def region_grow_by_depth(seed_points, depth_image, max_diff):
    """
    区域生长：以seed_points为种子，向外扩张，提取深度相差不大的区域
    :param seed_points: [(x1, y1), (x2, y2), ...]，顺序连成线段
    :param depth_image: 深度图，np.ndarray
    :param max_diff: 最大允许的深度差
    :return: mask，np.uint8，提取区域为255，其余为0
    """
    h, w = depth_image.shape
    mask = np.zeros((h, w), np.uint8)
    visited = np.zeros((h, w), np.bool_)
    queue = []

    # 计算种子点平均深度
    seed_depths = []
    for x, y in seed_points:
        if 0 <= x < w and 0 <= y < h and depth_image[y, x] > 0:
            seed_depths.append(depth_image[y, x])
            mask[y, x] = 255
            queue.append((x, y))
            visited[y, x] = True
    if not seed_depths:
        return mask
    mean_depth = np.mean(seed_depths)

    # 8邻域生长
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while queue:
        x, y = queue.pop(0)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                d = depth_image[ny, nx]
                if d > 0 and abs(int(d) - mean_depth) <= max_diff:
                    mask[ny, nx] = 255
                    queue.append((nx, ny))
                visited[ny, nx] = True
    return mask

def find_valid_square(pt, depth, h, w, size=5, step=2, min_ratio=0.5, max_iter=20):
    """在depth图上以pt为中心找一个有效深度占比>=min_ratio的正方形区域，否则朝有效率最大方向移动"""
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