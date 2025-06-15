'''
圆柱体姿态估计
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import os
import copy
import numpy as np
import cv2
import open3d as o3d
# 阿凯机器人工具箱
from kyle_robot_toolbox.open3d import *
from kyle_robot_toolbox.transform import *

def create_cylinder_mesh(cylinder_radius, cylinder_length):
    '''创建圆柱体Mesh'''
    # 创建圆柱体Mesh
    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(cylinder_radius, cylinder_length, \
                                                              resolution=100)
    cylinder_mesh.compute_vertex_normals()
    cylinder_mesh.compute_triangle_normals()
    # 变换
    cylinder_mesh.transform(Transform.rymat(np.pi/2))
    return cylinder_mesh

def cylinder_mesh2pcd(cylinder_mesh, voxel_size):
    '''将圆柱体的Mesh转换为PCD'''
    # 点云采样
    cylinder_pcd = cylinder_mesh.sample_points_uniformly(100000)
    cylinder_pcd = cylinder_pcd.voxel_down_sample(voxel_size=voxel_size)
    return cylinder_pcd

def get_half_cylinder_pcd(cylinder_pcd):
    '''获取半个圆柱体的PCD(只保留Z轴上方的)'''
    # 移除Z轴下方的点云
    cylinder_points = np.float32(cylinder_pcd.points)
    index = np.where(cylinder_points[:, 2] >= 0)[0]
    cylinder_half_pcd = cylinder_pcd.select_by_index(index)
    return cylinder_half_pcd

def preprocess_cylinder_pcd(obj_pcd, voxel_size=0.002, \
        normal_estimate_radius=0.01, \
        normal_estimate_max_nn=20, \
        is_debug=False):
    '''获取圆柱体的PCD点云，并做预处理'''
    # 获取待移除噪点的点的索引
    cl, ind = obj_pcd.remove_statistical_outlier(nb_neighbors=100,
                                         std_ratio=1.0)
    if is_debug:
        # 可视化
        display_inlier_outlier(obj_pcd, ind)
    # 移除噪点
    obj_pcd = obj_pcd.select_by_index(ind)
    # 点云下采样
    obj_pcd = obj_pcd.voxel_down_sample(voxel_size=voxel_size)
    # 法向量估计
    obj_pcd.estimate_normals(search_param=\
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_estimate_radius,
                max_nn=normal_estimate_max_nn))
    # 法向量重定向
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(\
                        obj_pcd, camera_location=[0,0,0])
    if is_debug:
        # 可视化
        draw_geometry([obj_pcd], window_name="移除离群点后的上表面点云(下采样后)")
    return obj_pcd


def draw_pen_pose(pcd_scene, cylinder_pcd, T_cam2obj):
    # 在场景中绘制笔的姿态
    cylinder_measure_pcd = copy.deepcopy(cylinder_pcd)
    cylinder_measure_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    cylinder_measure_pcd.transform(T_cam2obj)
    axis_obj = geometry.geometry_coordinate(T_cam2obj, size=0.05)
    axis_cam = geometry.geometry_coordinate(np.eye(4), size=0.05)
    draw_geometry([axis_cam, axis_obj, pcd_scene, cylinder_measure_pcd])

def draw_cylinder_pose(pcd_scene, cylinder_pcd, T_cam2obj,\
        size=0.05, color=[0.5, 0.5, 0.5], is_draw=True):
    '''在场景中绘制圆柱体的姿态'''
    cylinder_measure_pcd = copy.deepcopy(cylinder_pcd)
    cylinder_measure_pcd.paint_uniform_color(color)
    cylinder_measure_pcd.transform(T_cam2obj)
    axis_obj = geometry.geometry_coordinate(T_cam2obj, size=size)
    axis_cam = geometry.geometry_coordinate(np.eye(4), size=size)

    geometry_list = [axis_cam, axis_obj, pcd_scene, cylinder_measure_pcd]
    if is_draw:
        draw_geometry(geometry_list, window_name="场景点云+圆柱体")
    return geometry_list


def draw_cylinder_icp_result(target_pcd, source_pcd, T_target2source):
    '''展示ICP点云配准的结果'''
    source_pcd_trans = copy.deepcopy(source_pcd)
    source_pcd_trans.transform(T_target2source)
    source_pcd_trans.paint_uniform_color([0, 1, 0])
    # 创建坐标系
    axis_cam = geometry.geometry_coordinate(np.eye(4), size=0.05)
    axis_obj = geometry.geometry_coordinate(T_target2source, size=0.05)
    draw_geometry([axis_cam, axis_obj, target_pcd, source_pcd_trans],\
        window_name="点云配准结果")

def cylinder_icp(cylinder_half_pcd, obj_pcd, voxel_size, \
        T_cam2obj_init=None, \
        is_debug=False):
    '''圆柱体ICP配准'''
    # 使用一个比较小的阈值 - max_correspondence_distance_fine
    source_pcd = cylinder_half_pcd # 模板点云
    target_pcd = obj_pcd # 拍摄到的物体点云
    # 生成一个初始姿态
    if T_cam2obj_init is None:
        T_cam2obj_init = np.eye(4)
        # 中心点选择为点云中心点
        T_cam2obj_init[:3, 3] = obj_pcd.get_center()
        # Z轴初始化
        T_cam2obj_init[:3, :3] = Transform.rymat(np.pi, ndim=3)
    # 粗配准
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,5.0*voxel_size, T_cam2obj_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # 细配准
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, 1.5*voxel_size, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # 物体在相机坐标系下的姿态
    T_cam2obj_icp = icp_fine.transformation
    
    if is_debug:
        draw_cylinder_icp_result(target_pcd, source_pcd, T_cam2obj_icp)
    return T_cam2obj_icp

def get_angle_with_y_neg_2d(obj_x_unit, is_debug=False):
    '''获取向量投影在XY平面上，跟相机坐标系y轴负方向单位向量夹角'''
    obj_x_unit_2d = obj_x_unit[:2]
    y_neg_2d = np.float32([0, -1])
    theta = np.arccos(np.dot(y_neg_2d, obj_x_unit_2d))
    if is_debug:
        print(f"夹角: {np.degrees(theta):.1f} °")
    return theta

def cylinder_pose_refine(T_cam2obj_icp, is_debug=False):
    '''圆柱体姿态矫正'''
    # 在相机坐标系下, 物体坐标系的X轴单位向量
    obj_x_unit = T_cam2obj_icp[:3, 0]
    if is_debug:
        print(f"obj_x_unit = {obj_x_unit}")
    theta = get_angle_with_y_neg_2d(obj_x_unit, is_debug=True)
    if np.abs(theta) > np.pi/2:
        # 需要调整方向
        if is_debug:
            print("调整obj_x_unit的方向")
        obj_x_unit = -1*obj_x_unit
        theta = get_angle_with_y_neg_2d(obj_x_unit, is_debug=True)
    # 物体X轴单位向量在XY平面上投影
    # 将其也转化为单位向量
    obj_x_unit_2d = obj_x_unit[:2]
    obj_x_unit_init = np.float32([0, 0, 0])
    obj_x_unit_init[:2] = obj_x_unit_2d / np.linalg.norm(obj_x_unit_2d)
    # 计算转轴
    n = np.cross(obj_x_unit_init, obj_x_unit)
    # 获取旋转角度(obj_x_unit_init -> obj_x_unit)
    theta = np.arccos(np.dot(obj_x_unit_init, obj_x_unit))
    # 转换为旋转矩阵
    rmat = Transform.rvect2rmat(u=n, theta=theta)
    # 将旋转添加到Z轴上(初始时，在相机坐标系下的Z轴是负的)
    obj_z_unit_init = np.float32([0, 0, -1]).reshape((-1, 1))
    obj_z_unit = rmat @ obj_z_unit_init
    obj_z_unit = obj_z_unit.reshape(-1)
    # 通过叉乘计算Y轴
    obj_y_unit = np.cross(obj_z_unit, obj_x_unit)
    # 生成旋转矩阵
    R_cam2obj_refine = np.zeros((3, 3))
    R_cam2obj_refine[:, 0] = obj_x_unit
    R_cam2obj_refine[:, 1] = obj_y_unit
    R_cam2obj_refine[:, 2] = obj_z_unit
    # 中心点选择为点云中心点
    T_cam2obj_refine = np.copy(T_cam2obj_icp)
    # 赋值旋转矩阵
    T_cam2obj_refine[:3, :3] = R_cam2obj_refine
    # 位置跟之前保持不变
    if is_debug:
        print(f"T_cam2obj_refine = \n{T_cam2obj_refine}")
    return T_cam2obj_refine