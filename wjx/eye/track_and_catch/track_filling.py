# 功能说明
# 这个代码实现了以下功能：

# 路径点填充：

# 使用线性插值在相邻路径点之间添加额外点
# 确保相邻两点之间的距离小于指定的最大值
# 针对角度坐标特别处理，避免从359度到1度这类情况产生异常旋转
# ArUco位置文件解析：

# 解析您提供的 aruco_positions.txt 格式
# 过滤掉"未检测到标签"的行
# 提取每个检测点的X、Y、Z坐标
# 路径可视化：

# 使用matplotlib生成3D路径图
# 对比显示原始路径和填充后的路径
# 不同颜色和标记区分原始点和插值点
# 路径保存：

# 将填充后的路径点保存为文本文件
# 支持保存完整的6D位姿（包括姿态角度）
# 使用方法
# 保存上述代码为 path_filler.py
# 运行：
# 按提示输入最大允许距离
# 查看生成的可视化图表
# 选择是否保存填充后的路径
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import splprep, splev
import numpy as np

def sparsify_waypoints(waypoints, r):
    """
    稀疏化路径点：保留首尾点，且相邻点间距离大于r
    参数:
        waypoints: 原始路径点列表，每个点为[x, y, z]或[x, y, z, ...]
        r: 最小间隔距离
    返回:
        sparse_points: 稀疏化后的路径点列表
    """
    if len(waypoints) < 2:
        return waypoints.copy()
    sparse_points = [waypoints[0]]
    last_point = np.array(waypoints[0])
    for point in waypoints[1:-1]:
        point_arr = np.array(point[:3])
        if np.linalg.norm(point_arr - last_point[:3]) >= r:
            sparse_points.append(point)
            last_point = point_arr
    sparse_points.append(waypoints[-1])
    return sparse_points


def spline_interpolate_path(waypoints, num_points=200, s=0):
    """
    三维路径点样条插值，保证首尾点严格经过且头尾平滑
    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z]
        num_points: 插值后路径点数量
        s: 平滑因子，0为严格插值，适当增大可平滑
    返回:
        interp_points: 插值后的路径点列表
    """
    waypoints = np.array(waypoints)
    x, y, z = waypoints[:,0], waypoints[:,1], waypoints[:,2]
    # 用累积弦长参数化，保证首尾点插值自然
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    u = np.zeros(len(x))
    u[1:] = np.cumsum(distances)
    u /= u[-1]  # 归一化到[0,1]
    # 样条插值
    tck, _ = splprep([x, y, z], u=u, s=s)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new, z_new = splev(u_new, tck)
    interp_points = np.stack([x_new, y_new, z_new], axis=1)
    # 强制首尾点等于原始首尾点
    interp_points[0] = waypoints[0]
    interp_points[-1] = waypoints[-1]
    return interp_points.tolist()


def fill_path_points(waypoints, max_distance=10.0):
    """
    对路径点进行填充，使相邻两点之间的距离小于指定值
    
    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z, rx, ry, rz] 或 [x, y, z]
        max_distance: 允许的最大距离
        
    返回:
        filled_waypoints: 填充后的路径点列表
    """
    if len(waypoints) < 2:
        return waypoints.copy()
        
    # 确定点的维度
    point_dim = len(waypoints[0])
    
    filled_waypoints = [waypoints[0]]  # 从第一个点开始
    
    # 遍历每两个相邻点
    for i in range(1, len(waypoints)):
        # 获取前一个点和当前点
        prev_point = np.array(waypoints[i-1])
        current_point = np.array(waypoints[i])
        
        # 计算两点之间的欧氏距离（仅考虑位置部分）
        distance = np.linalg.norm(current_point[:3] - prev_point[:3])
        
        if distance <= max_distance:
            # 如果距离已经满足要求，直接添加当前点
            filled_waypoints.append(waypoints[i])
        else:
            # 需要在两点之间插入额外的点
            # 计算需要插入的点数
            num_points = int(np.ceil(distance / max_distance))
            
            # 对每个维度进行线性插值
            for j in range(1, num_points):
                # 插值比例
                ratio = j / num_points
                
                # 对每个维度进行插值
                # 对于位置坐标，使用线性插值
                # 对于角度坐标，需要考虑角度的特殊性
                interpolated_point = []
                
                for dim in range(point_dim):
                    if dim < 3:
                        # 位置坐标的线性插值
                        value = prev_point[dim] + ratio * (current_point[dim] - prev_point[dim])
                    else:
                        # 角度坐标的插值
                        angle1 = prev_point[dim]
                        angle2 = current_point[dim]
                        
                        # 处理角度差超过180度的情况
                        diff = angle2 - angle1
                        if abs(diff) > 180:
                            if diff > 0:
                                angle1 += 360
                            else:
                                angle2 += 360
                        
                        # 插值
                        value = angle1 + ratio * (angle2 - angle1)
                        
                        # 规范化到0-360度
                        while value >= 360:
                            value -= 360
                        while value < 0:
                            value += 360
                    
                    interpolated_point.append(value)
                
                filled_waypoints.append(interpolated_point)
            
            # 添加当前点
            filled_waypoints.append(waypoints[i])
    
    return filled_waypoints

def fill_path_points2(waypoints, max_distance=10.0, max_angle=10.0):
    """
    对路径点进行填充，使相邻两点之间的距离和角度变化都小于指定值
    不处理角度跳变，角度直接线性插值

    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z, rx, ry, rz] 或 [x, y, z]
        max_distance: 允许的最大距离
        max_angle: 允许的最大角度变化（度），仅对有姿态的点有效

    返回:
        filled_waypoints: 填充后的路径点列表
    """
    if len(waypoints) < 2:
        return waypoints.copy()
        
    point_dim = len(waypoints[0])
    filled_waypoints = [waypoints[0]]

    for i in range(1, len(waypoints)):
        prev_point = np.array(waypoints[i-1])
        current_point = np.array(waypoints[i])

        # 计算位置距离
        distance = np.linalg.norm(current_point[:3] - prev_point[:3])

        # 计算角度变化（仅对有姿态的点）
        angle_steps = 1
        if point_dim >= 6:
            angle_diffs = []
            for dim in range(3, 6):
                a1 = prev_point[dim]
                a2 = current_point[dim]
                diff = abs(a2 - a1)
                angle_diffs.append(diff)
            max_angle_diff = max(angle_diffs)
            angle_steps = int(np.ceil(max_angle_diff / max_angle))

        # 需要插值的步数
        steps = max(int(np.ceil(distance / max_distance)), angle_steps)

        if steps == 1:
            filled_waypoints.append(waypoints[i])
        else:
            for j in range(1, steps):
                ratio = j / steps
                interpolated_point = []
                for dim in range(point_dim):
                    value = prev_point[dim] + ratio * (current_point[dim] - prev_point[dim])
                    interpolated_point.append(value)
                filled_waypoints.append(interpolated_point)
            filled_waypoints.append(waypoints[i])

    return filled_waypoints

import csv

def read_xyz_from_csv(filename):
    """
    读取以 world_x,world_y,world_z 为表头的CSV文件，返回 [[x, y, z], ...]
    """
    waypoints = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            try:
                x, y, z = map(float, parts)
                waypoints.append([x, y, z])
            except ValueError:
                continue
    return waypoints

def read_xyzrxryrz_from_csv(filename):
    """
    读取以 X,Y,Z,RX,RY,RZ 为表头的CSV文件，返回 [[x, y, z, rx, ry, rz], ...]
    """
    waypoints = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            try:
                x, y, z, rx, ry, rz = map(float, parts)
                waypoints.append([x, y, z, rx, ry, rz])
            except ValueError:
                continue
    return waypoints

def spline_interpolate_6d_path(waypoints, num_points=200, s=0):
    """
    六维路径点样条插值，适用于[x, y, z, rx, ry, rz]
    只做样条插值，保证首尾点严格等于原始首尾点，不做角度归一化或特殊处理
    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z, rx, ry, rz]
        num_points: 插值后路径点数量
        s: 平滑因子，0为严格插值，适当增大可平滑
    """
    waypoints = np.array(waypoints)
    n_dim = waypoints.shape[1]
    interp_data = [waypoints[:, i] for i in range(n_dim)]
    # 用累积弦长参数化
    xyz = waypoints[:, :3]
    distances = np.sqrt(np.sum(np.diff(xyz, axis=0)**2, axis=1))
    u = np.zeros(len(xyz))
    u[1:] = np.cumsum(distances)
    u /= u[-1]
    # 样条插值
    tck, _ = splprep(interp_data, u=u, s=s)
    u_new = np.linspace(0, 1, num_points)
    interp_cols = splev(u_new, tck)
    interp_points = np.stack(interp_cols, axis=1)
    # 保证首尾点严格等于原始首尾点
    interp_points[0] = waypoints[0]
    interp_points[-1] = waypoints[-1]
    return interp_points.tolist()

def process_waypoints(waypoints, max_distance=0.3):
    """
    
    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z] 或 [x, y, z, rx, ry, rz]
        max_distance: 相邻两点之间的最大距离
        
    返回:
        filled_waypoints: 填充后的路径点列表
    """
    # 稀疏化路径点
    # sparse_points = sparsify_waypoints(waypoints, r=20)
    sparse_points = waypoints.copy()
    # 三维样条插值
    interp_points = spline_interpolate_6d_path(sparse_points, num_points=100, s=500)
    
    # 填充路径点
    filled_waypoints = fill_path_points2(interp_points, max_distance=max_distance, max_angle=0.05)
    
    return filled_waypoints

def parse_aruco_file(filename):
    """
    解析ArUco位置文件
    
    参数:
        filename: ArUco位置文件路径
        
    返回:
        waypoints: 路径点列表，每个点为 [x, y, z]
    """
    waypoints = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if "未检测到标签" in line:
            continue
            
        if "X=" in line and "Y=" in line and "Z=" in line:
            # 提取坐标值
            parts = line.strip().split(", ")
            x = float(parts[0].split("=")[1])
            y = float(parts[1].split("=")[1])
            z = float(parts[2].split("=")[1])
            
            waypoints.append([x, y, z])
    
    return waypoints

def visualize_path(original_waypoints, filled_waypoints, title="路径点填充对比"):
    """
    可视化原始路径和填充后的路径
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    orig_x = [p[0] for p in original_waypoints]
    orig_y = [p[1] for p in original_waypoints]
    orig_z = [p[2] for p in original_waypoints]
    
    filled_x = [p[0] for p in filled_waypoints]
    filled_y = [p[1] for p in filled_waypoints]
    filled_z = [p[2] for p in filled_waypoints]
    
    # 绘制原始路径
    ax.plot(orig_x, orig_y, orig_z, 'ro-', markersize=8, label='original path points')
    
    # 绘制填充后的路径
    ax.plot(filled_x, filled_y, filled_z, 'b.-', markersize=4, label='filled path points')
    
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_zlabel('Z 坐标')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def save_waypoints(waypoints, filename, include_rotation=False):
    """
    保存路径点到CSV文件

    参数:
        waypoints: 路径点列表
        filename: 保存的文件名
        include_rotation: 是否包含旋转信息
    """
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if include_rotation and len(waypoints[0]) >= 6:
            writer.writerow(['X', 'Y', 'Z', 'RX', 'RY', 'RZ'])
            for point in waypoints:
                writer.writerow([f"{point[0]:.2f}", f"{point[1]:.2f}", f"{point[2]:.2f}",
                                 f"{point[3]:.2f}", f"{point[4]:.2f}", f"{point[5]:.2f}"])
        else:
            writer.writerow(['X', 'Y', 'Z'])
            for point in waypoints:
                writer.writerow([f"{point[0]:.2f}", f"{point[1]:.2f}", f"{point[2]:.2f}"])
    print(f"路径点已保存到 {filename}")

def complete_waypoints_with_orientation(waypoints, default_orientation=[180.0, 0.0, 0.0]):
    """
    为路径点添加姿态信息
    
    参数:
        waypoints: 原始路径点列表，每个点为 [x, y, z]
        default_orientation: 默认姿态 [rx, ry, rz]
        
    返回:
        complete_waypoints: 添加姿态后的路径点列表，每个点为 [x, y, z, rx, ry, rz]
    """
    complete_waypoints = []
    for point in waypoints:
        if len(point) == 3:
            # 如果只有位置信息，添加姿态
            complete_waypoints.append(point + default_orientation)
        else:
            # 已经有完整信息
            complete_waypoints.append(point)
    return complete_waypoints

if __name__ == "__main__":
    # 文件路径
    #aruco_file = '/home/hn/One-shot-imitation-in-chemistry-lab-main/eye/aruco_positions.txt'
    #aruco_file = "/home/hn/One-shot-imitation-in-chemistry-lab-main/eye/object_world_positions.txt"
    #aruco_file = "/home/xuan/dianrobot/wjx/eye/hand_positions.txt"
    #csv_file = "/home/xuan/dianrobot/wjx/eye/hand_positions.csv"
    csv_file = "/home/xuan/dianrobot/wjx/eye/track_and_catch/gripper_positions.csv"
    # 读取原始路径点
    #original_waypoints = parse_aruco_file(csv_file)
    #original_waypoints = read_xyz_from_csv(csv_file)
    original_waypoints = read_xyzrxryrz_from_csv(csv_file)
    print(f"读取到 {len(original_waypoints)} 个原始路径点")
    #original_waypoints[-1][1] = original_waypoints[-1][1] - 30 
    # 设置最大距离参数
    max_distance = 0.3
    
    # 填充路径点
    # original_waypoints = sparsify_waypoints(original_waypoints, r=20)
    # interp_points = spline_interpolate_path(original_waypoints, num_points=300, s=25)
    filled_waypoints = process_waypoints(original_waypoints, max_distance=max_distance)
    print(f"填充后共有 {len(filled_waypoints)} 个路径点")
    
    # 可视化对比
    visualize_path(original_waypoints, filled_waypoints)
    
    
    output_file = "/home/xuan/dianrobot/wjx/eye/filled.csv"
    save_waypoints(filled_waypoints, output_file, include_rotation=True)
    print("处理完成!")