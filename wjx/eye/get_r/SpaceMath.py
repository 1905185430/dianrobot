'''
通用的空间数学函数
包含将相机坐标系下的三维点转换为世界坐标系下的三维点的函数
transform_to_world_coordinates
'''
import numpy as np
import cv2



def Cam2World(cam2world, cam_point3d):
    """
    将相机坐标系下的三维点 cam_point3d 转换为世界坐标系下的三维点
    cam_point3d: [x, y, z]，相机坐标系下的点
    返回: [x_w, y_w, z_w]，世界坐标系下的点
    """
    # 获取相机到世界的旋转矩阵和平移向量
    # 假设 self.cam2world_R 为 3x3旋转矩阵，self.cam2world_T 为3x1平移向量
    cam2world = np.array(cam2world)
    cam2world_R = np.array(cam2world[:3, :3])
    cam2world_T = np.array(cam2world[:3, 3])    
    # 将相机坐标系下的点转换为齐次坐标
    cam_point3d_homogeneous = np.array(cam_point3d + [1.0])  # 添加齐次坐标
    # 进行坐标转换
    world_point_homogeneous = np.dot(cam2world_R, cam_point3d_homogeneous[:3]) + cam2world_T
    # 返回世界坐标系下的点
    return world_point_homogeneous.tolist()


def depth_pixel2cam_point3d(px, py, depth_image=None, intrinsic=None):
    """
    深度像素坐标(px, py)转换为相机坐标系下的三维坐标[x, y, z]
    参数:
        px, py: 像素坐标
        depth_image: 深度图（可选）
        depth_value: 深度值（可选，单位与深度图一致）
        intrinsic: 相机内参矩阵（可选，3x3）
    返回:
        [x_cam, y_cam, z_cam]: 相机坐标系下三维坐标
    """
    # 获取深度值
    if depth_image is not None:
        z_cam = depth_image[py, px]
    else:
        raise ValueError("必须提供 depth_image ")
    # 获取内参
    if intrinsic is not None:
        intrinsic = np.array(intrinsic)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
    else:
        raise ValueError("必须提供相机内参 intrinsic")
    # 计算相机坐标
    x_cam = (px - cx) * z_cam / fx
    y_cam = (py - cy) * z_cam / fy
    return [x_cam, y_cam, z_cam]


def fit_plane(points):
    """
    拟合3D点集的平面，返回平面方程参数 (a, b, c, d)，即 ax+by+cz+d=0
    points: N x 3 的numpy数组或列表
    """
    points = np.array(points)
    # 计算点集均值
    centroid = np.mean(points, axis=0)
    # 去中心化
    centered = points - centroid
    # SVD分解
    _, _, vh = np.linalg.svd(centered)
    # 法向量为最后一行
    normal = vh[-1]
    a, b, c = normal
    # 计算d
    d = -np.dot(normal, centroid)
    return a, b, c, d


def project_vector_on_plane(plane, vector):
    """
    输入：plane - 平面方程参数 [a, b, c, d]，vector - 需要投影的向量 [vx, vy, vz]
    输出：该向量在平面内的投影向量 [px, py, pz]
    """
    plane = np.array(plane)
    normal = plane[:3]
    normal = normal / np.linalg.norm(normal)
    vector = np.array(vector)
    # 向量在法向上的分量
    v_n = np.dot(vector, normal) * normal
    # 投影向量 = 原向量 - 法向分量
    proj = vector - v_n
    return proj.tolist()


def perp_vector_on_plane_with_ref(plane, a, b):
    """
    输入：
        plane - 平面方程参数 [a, b, c, d]
        a - 平面内的向量 [ax, ay, az]
        b - 参考向量 [bx, by, bz]
    输出：
        平面内与a垂直且与b夹角小于90度的单位向量
    """
    plane = np.array(plane)
    normal = plane[:3]
    normal = normal / np.linalg.norm(normal)
    a = np.array(a)
    b = np.array(b)
    # 先求a在平面内的单位向量
    a_proj = a - np.dot(a, normal) * normal
    if np.linalg.norm(a_proj) < 1e-8:
        raise ValueError("向量a与法向量平行，无法在平面内找到垂直向量")
    a_proj = a_proj / np.linalg.norm(a_proj)
    # 在平面内与a垂直的单位向量 = 法向量与a_proj的叉积
    perp = np.cross(normal, a_proj)
    perp = perp / np.linalg.norm(perp)
    # 判断与b的夹角是否小于90度
    if np.dot(perp, b) < 0:
        perp = -perp
    return perp.tolist()


def get_R_from_vectors(x_axis, y_axis, z_axis):
    """
    根据三个正交向量获取旋转矩阵
    输入：
        x_axis - x轴向量 [x1, y1, z1]
        y_axis - y轴向量 [x2, y2, z2]
        z_axis - z轴向量 [x3, y3, z3]
    输出：
        3x3的旋转矩阵
    """
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    z_axis = np.array(z_axis)
    
    # 确保向量是单位向量
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    return np.column_stack((x_axis, y_axis, z_axis))

def get_rxryrz_from_R(R):
    """
    从旋转矩阵R中提取绕x、y、z轴的旋转角度（单位：度）
    使用OpenCV库，rz范围为[-90, 90]度
    输入：
        R - 3x3的旋转矩阵
    输出：
        (rx, ry, rz) - 绕x、y、z轴的旋转角度（角度）
    """
    import numpy as np
    import cv2

    R = np.array(R)
    euler_angles, *_ = cv2.RQDecomp3x3(R)
    rx, ry, rz = euler_angles
    # 将rz限制在[-90, 90]度
    if rz > 50:
        rz = 50
    elif rz < -45:
        rz = -45

    # 将ry限制在[0, 180]度
    if ry < -30:
        ry = 0
    elif ry > 180:
        ry -= 180
    return rx, ry, rz