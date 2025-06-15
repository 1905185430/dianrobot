'''
Open3D手掌可视化GUI
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import numpy as np
import open3d as o3d

def get_landmark_color_list():
    '''获取手掌Landmark的颜色列表'''
    # 颜色列表
    color_list = np.zeros((21, 3), dtype=np.float64)
    # 红色索引
    red_idx = [0, 1, 5, 9, 13, 17]
    color_list[red_idx] = [1.0, 0, 0]
    # 黄色索引
    yellow_idx = [2, 3, 4]
    color_list[yellow_idx] = [1.0, 1.0, 0]
    # 紫色索引
    purple_idx = [6, 7, 8]
    color_list[purple_idx] = [1.0, 0, 1.0]
    # 橙色索引
    orange_idx = [10, 11, 12]
    color_list[orange_idx] = [1.0, 0.5, 0]
    # 绿色索引
    green_idx = [14, 15, 16]
    color_list[green_idx] = [0, 1.0, 0]
    # 蓝色索引
    blue_idx = [18, 19, 20]
    color_list[blue_idx] = [0, 0, 1.0]
    return color_list

def get_landmark_sphere(point3d_list):
    '''绘制手掌Landmark的球体'''
    # 获得颜色列表
    color_list = get_landmark_color_list()
    # 三维几何体列表
    geometry_list = []
    # 追加三维坐标
    for pidx, point in enumerate(point3d_list):
        # 获取三维坐标
        position = point.reshape((3, 1))
        # 创建三维球体
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(\
                        radius=0.005, resolution=10)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color_list[pidx])
        mesh_sphere.translate(position)
        # 追加Mesh
        geometry_list.append(mesh_sphere)
    return geometry_list

def get_hand_lineset(point3d_list):
    '''获取手掌线条几何体'''
    # 创建线条与颜色列表
    line_list = []
    line_color_list = []
    # - 添加基座线条
    base_lines =  [[0, 1], [0, 5], [0, 17], \
            [5, 9], [9, 13], [13, 17]]
    line_list += base_lines
    line_color_list += [[0.5, 0.5, 0.5] for i in range(len(base_lines))]
    # - 添加拇指线条
    fig1_lines = [[1, 2], [2, 3], [3, 4]]
    line_list += fig1_lines
    line_color_list += [[1.0, 1.0, 0.0] for i in range(len(fig1_lines))]
    # - 添加食指线条
    fig2_line = [[5, 6], [6, 7], [7, 8]]
    line_list += fig2_line
    line_color_list += [[1.0, 0, 1.0] for i in range(len(fig2_line))]
    # - 添加中指线条
    fig3_line = [[9, 10], [10, 11], [11, 12]]
    line_list += fig3_line
    line_color_list += [[1.0, 0.5, 0] for i in range(len(fig3_line))]
    # - 添加无名指线条
    fig4_line = [[13, 14], [14, 15], [15, 16]]
    line_list += fig4_line
    line_color_list += [[0, 1.0, 0] for i in range(len(fig3_line))]
    # - 添加小拇指线条
    fig5_line = [[17, 18], [18, 19], [19, 20]]
    line_list += fig5_line
    line_color_list += [[0, 0, 1.0] for i in range(len(fig3_line))]
    # 绘制线段
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(line_list)
    lines_pcd.colors = o3d.utility.Vector3dVector(line_color_list) #线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(point3d_list)
    return lines_pcd

def get_hand_geomery(point3d_list):
    '''绘制手部姿态'''
    # 添加小球
    geometry_list = get_landmark_sphere(point3d_list)
    # 添加线段
    lines_pcd = get_hand_lineset(point3d_list)
    geometry_list.append(lines_pcd)
    # 添加坐标系
    # 绘制手掌坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, \
                                        origin=[0, 0, 0])
    geometry_list.append(axis)
    return geometry_list