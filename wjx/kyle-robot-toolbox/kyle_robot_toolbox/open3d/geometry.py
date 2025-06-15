'''
Open3D 可视化工具库
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np
import cv2
import open3d as o3d
from kyle_robot_toolbox.transform import *

###########################################
#  可视化相关
###########################################
def draw_geometry(geometry_list, bk_color=[1.0, 1.0, 1.0], \
				point_show_normal=False, light_on=True, \
	  			window_name="Open3D"):
	'''绘制3D几何体'''
	# 创建可视化器
	vis = o3d.visualization.Visualizer()
	# 创建窗口
	vis.create_window(window_name=window_name)
	# 配置属性
	opt = vis.get_render_option()
	# - 设置背景颜色 -> 黑色
	opt.background_color = np.asarray(bk_color)
	opt.point_show_normal = point_show_normal
	# - 打开灯光
	opt.light_on = light_on
	# 添加Geomery
	for geometry in geometry_list:
		# vis.add_geometry({"geometry": geometry})
		vis.add_geometry(geometry)
	# 显示窗口
	vis.run()
	# 销毁窗口
	vis.destroy_window()

def geometry_coordinate(T, size=0.1):
	'''生成坐标系
	@T: 空间变换矩阵4x4
	@size: 坐标轴的长度
	'''
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
	frame.transform(T)
	return frame

def geometry_points(point_list, radius=0.01, color=[1.0, 0, 0]):
    '''绘制3D点
    @point_list: 3D点的列表
    @radius: 小球半径
    @color: 小球颜色, 默认红色
    '''
    if color is None:
        color = [0.8, 0.2, 0.8]
    geometries = []
    for pt in point_list:
        # 创建小球
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                            resolution=20)
        # 小球平移
        sphere.translate(pt)
        # 绘制统一的颜色
        sphere.paint_uniform_color(np.array(color))
        # 追加到Geometry列表里
        geometries.append(sphere)
    return geometries

# def geometry_points(point_list, radius=0.01, color=[1.0, 0, 0]):
# 	'''绘制3D点
# 	@point_list: 3D点的列表
# 	@radius: 小球半径
# 	@color: 小球颜色, 默认红色
# 	'''
# 	if color is None:
# 		color = [0.8, 0.2, 0.8]
# 	geometries = []
# 	for pt in point_lists:
# 		# 创建小球
# 		sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
# 															resolution=20)
# 		# 小球平移
# 		sphere.translate(pt)
# 		# 绘制统一的颜色
# 		sphere.paint_uniform_color(np.array(color))
# 		# 追加到Geometry列表里
# 		geometries.append(sphere)
# 	return geometries

def geometry_camera(intrinsic, T_world2cam, \
		img_width, img_height, \
		panel_distance = 0.1, \
		color=[1.0, 0.0, 0.0], \
      	draw_panel=False):
	'''
 	@intrinsic: 相机内参
	@T_world2cam: 相机外参 T_world2cam
		相机在世界坐标系下的位姿
	@img_width: 图像宽度
	@img_height: 图像高度
	@panel_distance: 相机模型原点到可视化平面的距离
			决定了相机可视化模型的尺寸
	@color: 线条颜色
  	'''
	# 提取相机内参矩阵的元素
	fx = intrinsic[0, 0]
	fy = intrinsic[1, 1]
	cx = intrinsic[0, 2]
	cy = intrinsic[1, 2]
	# 将外参转换为旋转与平移
 
	R = T_world2cam[:3, :3]
	t = T_world2cam[:3, 3]
	# 点在相机坐标系下的Z轴坐标, 单位m
	Z_i = panel_distance
	# 相机内参
	K = np.array([
	 		[fx, 0, cx],
			[0, fx, cy],
			[0, 0, 1]])
	# 计算相机内参的逆
	K_inv = np.linalg.inv(K)

	# 生成相机坐标系
	# 坐标轴长度为Z_i/2
	coord_cam = geometry_coordinate(T_world2cam, size=0.5*Z_i)
	# 像素坐标系下的点
	# - 坐标系原点 + 图像的四个角点, Z轴归一化
	points_pixel = [
		[0, 0, 0],					# 0 原点
		[0, 0, 1], 					# 1 图像左上角
		[img_width, 0, 1], 			# 2 图像的右上角
		[0, img_height, 1],			# 3 图像左下角
		[img_width, img_height, 1], # 4 图像右下角
	]
	# 将像素转换为相机坐标系中的点
	# Z_i * UV_i =  K * ^{cam}P
	# ^{cam}P = Z_i * K^{-1} *  UV_i
	# 单位m
	points_m = np.float32([Z_i * K_inv @ p for p in points_pixel])
	# # 单位从mm转换为m
	# points_m = 0.001 * points_mm
	if draw_panel:
		# 绘制图像平面
		# - 计算平面的宽度
		# 	平面左上角点的X坐标 减去图像右下角点X坐标
		width = abs(points_m[1][0] - points_m[4][0])
		# - 计算平面的高度
		# 	平面左上角点的Y坐标 减去图像右下角点Y坐标
		height = abs(points_m[1][1] - points_m[4][1])
		print(f"实际宽度: {width} 实际高度: {height} 单位m")
		# - 创建一个立方体， 厚度很薄
		box_depth = 1e-6
		plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=box_depth)
		plane.paint_uniform_color(color)
		# 先平移到相机坐标系原点
		plane.transform(T_world2cam)
		# 平移BOX 非对称
		plane.translate(R @ [points_m[1][0], points_m[1][1], Z_i])
		# plane.translate(R @ [-width/2, -height/2, Z_i-box_depth])

	
	# pyramid
	# 绘制金字塔 - 由直线构成
	# R @ p + t 将点从相机坐标系转换为世界坐标系
	points_in_world = [(R @ p + t) for p in points_m]
	# lines存放的是点与点之间的连线关系
	# 例如[0, 1] 所代表的是第0个点与第1个点组成一条连线。
	if draw_panel:
		lines = [
			[0, 1],
			[0, 2],
			[0, 3],
			[0, 4],
		]
	else:
		# 没有平面要多加几条线
		lines = [
			[0, 1],
			[0, 2],
			[0, 3],
			[0, 4],
			[1, 2],
			[2, 4],
			[4, 3],
			[3, 1], 
		]
	# 绘制连线, 颜色统一用color
	colors = [color for i in range(len(lines))]
	fov_line_set = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(points_in_world),
		lines=o3d.utility.Vector2iVector(lines))
	fov_line_set.colors = o3d.utility.Vector3dVector(colors)
	
	# 返回Geometry
	if draw_panel:
		return [coord_cam, plane, fov_line_set]
	else:
		return [coord_cam, fov_line_set]


def geometry_box(T_world2box, width, height, \
    	box_depth=0.01, color=[0.5, 0.5, 0.5]):
	'''绘制盒子
 	Open3D里面的盒子默认在盒子的一个角角上。
	默认将坐标系原点定义在盒子上表面
  	'''
	box = o3d.geometry.TriangleMesh.create_box(width, height, depth=box_depth)
	box.paint_uniform_color(color)
	# 先平移到相机坐标系原点
	box.transform(T_world2box)
	# 盒子后退，移动到它该到的地方
	box.translate(T_world2box[:3, :3] @ [-width/2, -height/2, -box_depth])
	return box

def display_inlier_outlier(cloud, ind):
	'''显示内点与离群点
	红色: 离群点
	内点: 灰色
	'''
	inlier_cloud = cloud.select_by_index(ind)
	outlier_cloud = cloud.select_by_index(ind, invert=True)
	outlier_cloud.paint_uniform_color([1, 0, 0])
	inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
	draw_geometry([inlier_cloud, outlier_cloud])

def geometry_norm_arrow(center_point, center_normal_vector, \
		sphere_radius=0.003, sphere_color=[1.0, 1.0, 1.0],\
		arrow_color=[0.0, 0.0, 1.0], is_debug=False):
    '''获取法向量可视化模型'''
    # 可视化，在质心位置绘制小球
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    mesh_sphere.compute_vertex_normals()
    # 小球平移
    mesh_sphere.translate(center_point.reshape(-1), relative=False)
    # 给小球上色为红色
    mesh_sphere.paint_uniform_color(sphere_color)
    
    # 计算向量之间的夹角
    z0 = np.float64([0, 0, 1]).reshape((3, 1))
    cos_theta = z0.T.dot(center_normal_vector)
    theta = np.arccos(cos_theta)
    # 向量叉乘得到旋转轴
    rot_vect = np.cross(z0.reshape(-1), center_normal_vector.reshape(-1))
    rot_vect /= np.linalg.norm(rot_vect) # 归一化
    if is_debug:
        print(f"旋转向量: {rot_vect}")
    # 构造旋转矩阵
    rot_mat = cv2.Rodrigues(rot_vect*theta)[0]
    if is_debug:
        print(f"旋转矩阵:\n  {rot_mat}")
    # 绘制箭头
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(\
                        cylinder_radius=0.002, cone_radius=0.004,\
                        cylinder_height=0.05,  cone_height=0.01, \
                        resolution=20, cylinder_split=4, cone_split=1)

    mesh_arrow.paint_uniform_color(arrow_color)
    mesh_arrow.rotate(rot_mat, center=(0, 0, 0))
    mesh_arrow.translate(center_point.reshape(-1), relative=True)
    return mesh_sphere, mesh_arrow


def draw_quaternion_trajectory(q_waypoint_list, q_list):
    '''绘制四元数的轨迹, 展示Squad的结果'''
    z_waypoint_list = Quaternion.get_z_list(q_waypoint_list)
    z_list = Quaternion.get_z_list(q_list)
    # 途经点为了方便可视化，做了一点点的偏移量
    waypoints_list = geometry_points(z_waypoint_list, radius=0.0085, color=[1, 0, 0])
    for point in waypoints_list:
        point.compute_vertex_normals()
    points_list = geometry_points(z_list, radius=0.004, color=[0, 1, 0])
    for point in points_list:
        point.compute_vertex_normals()

    # 绘制中心点到姿态
    lineset_color = [0, 0, 1]
    lineset_points = z_list.tolist()
    # lineset_points.insert(0, [0, 0, 0])
    lineset_vect = [[i, i-1] for i in range(1, len(z_list))]
    # 绘制四元数前进轨迹
    lineset_colors = [lineset_color for i in range(len(lineset_vect))]
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lineset_points),
        lines=o3d.utility.Vector2iVector(lineset_vect))
    lineset.colors = o3d.utility.Vector3dVector(lineset_colors)
    # 绘制Shpere
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.98, resolution=100)
    center_sphere.paint_uniform_color(np.array([0.8, 0.8, 0.8]))
    center_sphere.compute_vertex_normals()
    # 绘制中心点到
    draw_geometry(waypoints_list + points_list + [lineset, center_sphere])
