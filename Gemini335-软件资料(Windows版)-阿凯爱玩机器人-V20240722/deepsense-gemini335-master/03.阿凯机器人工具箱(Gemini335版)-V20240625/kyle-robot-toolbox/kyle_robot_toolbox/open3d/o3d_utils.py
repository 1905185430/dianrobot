'''
Open3D常用工具函数
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import copy
import numpy as np
import cv2
import open3d as o3d
import trimesh
from kyle_robot_toolbox.transform import *

def load_tri_mesh_list(mesh_path):
	'''载入TMesh列表'''
	# 读取.obj文件
	tmesh = trimesh.load_mesh(mesh_path)
	# 提取列表形式的Mesh
	tmesh_list = None
	is_single_mesh = (type(tmesh) == trimesh.base.Trimesh)
	if is_single_mesh:
		# 如果只有一个，就是三角面片Trimesh
		# trimesh.base.Trimesh
		tmesh_list = [tmesh]
	else:
		# 场景Scene, 包含多个三角面片
		# trimesh.scene.scene.Scene
		tmesh_list = list(tmesh.geometry.values())
	return tmesh_list

def tri_mesh2o3d_mesh(tmesh):
	'''将Trimesh转换为Open3D格式Mesh'''
	omesh = o3d.geometry.TriangleMesh()
	omesh.vertices = o3d.utility.Vector3dVector(np.asarray(tmesh.vertices))
	omesh.triangles = o3d.utility.Vector3iVector(np.asarray(tmesh.faces))
	omesh.compute_vertex_normals()
	mesh_color = np.float64(np.random.randint(0, 100, 3))/100.0
	omesh.paint_uniform_color(mesh_color)
	return omesh

def load_o3d_mesh_via_trimesh(mesh_path):
	'''通过Trimesh载入Open3D格式的Mesh(给子凸包随机上色)'''
	tmesh_list = load_tri_mesh_list(mesh_path)
	omesh_list = [tri_mesh2o3d_mesh(tmesh) for tmesh in tmesh_list]
	omesh = omesh_list[0]
	for i in range(1, len(omesh_list)):
		omesh += omesh_list[i]
	# 计算法向量
	omesh.compute_vertex_normals()
	return omesh

def load_o3d_mesh(mesh_path, color=[0, 0.5, 0.5]):
	'''载入Open3D的Mesh'''
	# 读取Mesh
	o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
	# 计算法向量
	o3d_mesh.compute_vertex_normals()
	# 绘制颜色
	o3d_mesh.paint_uniform_color(color)
	return o3d_mesh

def create_point_cloud(color_image, depth_image, camera_intrinsics, depth_scale=1):
	'''创建点云'''
	# 获取彩图尺寸
	height, width, _ = color_image.shape
	# 缩放深度图
	if depth_scale != 1.0:
		depth_image = depth_image / depth_scale
	# 得到索引号
	valid_index = depth_image != 0
	# 得到有效点云的RGB数值
	color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
	colors = color_rgb[valid_index].reshape((-1, 3)) / 255
	
	# 相机内参矩阵
	fx = camera_intrinsics[0, 0]
	fy = camera_intrinsics[1, 1]
	cx = camera_intrinsics[0, 2]
	cy = camera_intrinsics[1, 2]
	# 创建一个空的点云对象
	pcd = o3d.geometry.PointCloud()
	# 根据相机内参矩阵计算3D坐标
	py, px = np.indices((height, width))
	# 提取
	px_valid = px[valid_index]
	py_valid = py[valid_index]
	z = depth_image[valid_index]
	# 计算相机坐标系下的三维坐标
	x = (px_valid - cx) * z / fx
	y = (py_valid - cy) * z / fy
	points = np.stack([x, y, z], axis=-1)
	# 将3D坐标转换为点云对象
	points = points.reshape(-1, 3)
	# 填充PCD对象
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors)
	return pcd

def depth_img2canvas(depth_img, min_distance = 500,\
			max_distance = 2000):
	'''将深度图转换为画布'''
	# 距离阈值的单位是mm
	depth_img_cut = np.copy(depth_img).astype(np.float32)
	depth_img_cut[depth_img < min_distance] = min_distance
	depth_img_cut[depth_img > max_distance] = max_distance
	# 归一化
	depth_img_norm = np.uint8(255.0*(depth_img_cut-min_distance)/(max_distance - min_distance))
	# 转换为
	depth_colormap = cv2.applyColorMap(depth_img_norm, cv2.COLORMAP_JET)
	return depth_colormap	

def mesh_unit_mm2m(mesh_mm):
	'''三角Mesh单位从mm转换为m'''
	# 获得顶点 单位mm
	vertices_mm = np.asarray(mesh_mm.vertices)
	# 将mm转换为m
	vertices_m = vertices_mm / 1000.0
	mesh_m = copy.deepcopy(mesh_mm)
	mesh_m.vertices = o3d.utility.Vector3dVector(vertices_m)
	# 重新计算Mesh三角形面片的法向量
	mesh_m.compute_triangle_normals()
	return mesh_m

def remove_work_panel_points(scene_pcd, panel_model, distance_threshold=0.003):
	'''移除工作平面上的点
	@scene_pcd: 场景PCD
	@panel_model: 平面模型 [a, b, c, d]
	@distance_threshold: 距离阈值，距离目标平面距离小于这个值的也丢掉
	'''
	# 获取Numpy格式的点集
	points_3d = np.asarray(scene_pcd.points)
	# 分别获取X坐标, Y坐标, Z坐标的列表
	x_list = points_3d[:, 0]
	y_list = points_3d[:, 1]
	z_list = points_3d[:, 2]
	# 提取平面模型
	a, b, c, d = panel_model
	# 获取平面内侧的点云
	value = a*x_list + b*y_list + c*z_list + d
	# 获取工作平面内侧点集的索引
	if d > 0:
		pcd_close_panel_index = np.argwhere(value > distance_threshold)
	else:
		pcd_close_panel_index = np.argwhere(value < -distance_threshold)
	scene_pcd_close_panel = scene_pcd.select_by_index(pcd_close_panel_index)
	return scene_pcd_close_panel

def pcd_project2panel(pcd,A,B,C,D,x0,y0,z0):
	'''将PCD点云投影到空间平面'''
	source = copy.deepcopy(pcd)
	x1 = np.asarray(source.points)[:,0]
	y1 = np.asarray(source.points)[:,1]
	z1 = np.asarray(source.points)[:,2]
	x0 = x0 * np.ones(x1.size)
	y0 = y0 * np.ones(y1.size)
	z0 = z0 * np.ones(z1.size)
	r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5)
	a = (x1-x0)/r
	b = (y1-y0)/r
	c = (z1-z0)/r
	t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D)
	t = t / (a*A+b*B+c*C)
	np.asarray(source.points)[:,0] = x1 + a * t
	np.asarray(source.points)[:,1] = y1 + b * t
	np.asarray(source.points)[:,2] = z1 + c * t
	return source

def get_panel_normal_vector(a, b, c, d):
	'''根据平面方程获取平面法向量'''
	# 获取法向量
	n = np.float64([a, b, c])
	# 归一化
	n /= np.linalg.norm(n)
	return n

def normal_vector_redirect(n, obj_posi, \
		camera_posi=np.float64([0, 0, 0]).reshape(3, 1)):
	'''法向量重定向'''
	n = np.float64(n).reshape(3, 1)
	obj_posi = np.float64(obj_posi).reshape(3, 1)
	# 构造向量, obj_posi的反向, 单位向量
	v = -obj_posi
	v /= np.linalg.norm(v)
	# 利用点乘判断是否同向
	cos_theta = np.dot(v.reshape(-1), n.reshape(-1))
	if cos_theta < 0:
		# 夹角超过180度，需要反向
		n *= -1
	return n

def get_rmat_from_z(z, z0=None):
	if type(z) is not np.ndarray:
		z = np.float64(z).reshape((3, 1))
	# 计算向量之间的夹角
	if z0 is None:
		z0 = np.float64([0, 0, 1]).reshape((3, 1))
	cos_theta = z0.T.dot(z)
	theta = np.arccos(cos_theta)
	# print(f"cos_theta: {cos_theta} theta={np.degrees(theta)}")
	# 向量叉乘得到旋转轴
	rot_vect = np.cross(z0.reshape(-1), z.reshape(-1))
	rot_vect /= np.linalg.norm(rot_vect) # 归一化
	# print(f"旋转向量: {rot_vect}")
	# 构造旋转矩阵
	rot_mat = cv2.Rodrigues(rot_vect*theta)[0]
	# print(f"旋转矩阵:\n  {rot_mat}")
	return rot_mat

def get_distance_to_panel(a, b, c, d, point3d):
	'''计算中空间中的一个点距离平面的距离'''
	point3d = np.float64(point3d)
	px, py, pz = point3d.reshape(-1)
	distance = np.abs(a*px+b*py+c*pz+d)/np.sqrt(a**2 + b**2 + c**2)
	return distance

def point_to_plane_distance(panel_model, point3d):
	'''计算点到平面的距离'''
	a, b, c, d = panel_model
	x_i, y_i, z_i = point3d
	numerator = abs(a*x_i + b*y_i + c*z_i + d)
	denominator = np.sqrt(a**2 + b**2 + c**2)
	distance = numerator / denominator
	return distance

def line_plane_intersection(p1, p2, plane_expression):
	"""
	计算直线与平面的交点
	:param p1: 直线上的点P1
	:param p2: 直线上的点P2
	:param plane_expression: 空间平面的一般式方程 (a, b, c, d)
	:return: 直线与空间平面的交点坐标
	"""
	# 强制类型转换
	p1 = np.float64(p1)
	p2 = np.float64(p2)
	# 解析平面表达式
	a, b, c, d = plane_expression
	# 平面法向量
	plane_normal = np.float64([a, b, c])
	temp = np.sqrt(np.vdot(plane_normal, plane_normal))
	p1_d = np.vdot(p1,plane_normal)+d/temp
	p1_d2 = np.vdot(p2-p1,plane_normal)/temp
	n = p1_d2/p1_d
	p = p1 + n*(p2- p1)#所求交点
	return -p


def adjust_board_pose_by_pcd(T_cam2board, \
		pcd_roi,\
		distance_threshold = 0.002, \
		is_debug=False):
	'''通过PCD点云调整标定板位姿
	@T_cam2board: 标定板在相机坐标系下的位姿(单位mm)
	@pcd_roi: 标定板中心附近的点云样本
	@distance_threshold: 距离阈值, 距离平面多少的点被认为是在这个平面上的点
		距离阈值 单位m
	'''
	# 提取平移向量与旋转矩阵
	t_cam2board = T_cam2board[:3, 3]
	R_cam2board = T_cam2board[:3, :3]
	#####################################
	## 平面拟合
	#####################################
	# 平面拟合
	plane_model, inliers = pcd_roi.segment_plane(distance_threshold=distance_threshold,
											ransac_n=5,
											num_iterations=1000)
	# 打印拟合平面的表达式
	[a, b, c, d] = plane_model
	if is_debug:
		print(f"拟合桌面平面的表达式: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
	# 工作平面上的点云
	ws_panel_cloud = pcd_roi.select_by_index(inliers)
	#####################################
	## 位置矫正
	#####################################
	# - 计算从坐标系原点到t_cam2board构成的射线与平面的交点
	t_cam2board_m = np.float64(t_cam2board) / 1000.0
	t_cam2board_new_m = line_plane_intersection([0, 0, 0], t_cam2board_m, plane_model) 
	t_cam2board_new = t_cam2board_new_m * 1000.0
	if is_debug:
		print(f"矫正前t_cam2board: {t_cam2board}")
		print(f"矫正后t_cam2board_new: {t_cam2board_new}")
	# - 单位从m转换为mm
	# t_cam2board_new = [v*1000.0 for v in t_cam2board_new]
	#####################################
	## 姿态矫正
	#####################################
	# 获取基于2D位姿估计坐标系Z轴坐标
	rmat1 = R_cam2board
	z1 = rmat1[:, 2]
	# 获取平面法向量
	pannel_nvect = get_panel_normal_vector(a, b, c, d)
	# 法向量重定向
	pannel_nvect = normal_vector_redirect(pannel_nvect, obj_posi=t_cam2board)
	if is_debug:
		print(f"平面法向量: \n{pannel_nvect}")
	# z1跟pannel_nvect都是在世界坐标系(相机坐标系下的)
	# 在相机坐标系下， 从z1旋转到法向量pannel_nvect的位置
	rmat2 =  get_rmat_from_z(pannel_nvect, z0=z1)
	# 因此在矫正的时候， 需要左乘rmat2
	R_justify = np.dot(rmat2, rmat1)
	if is_debug:
		print(f"矫正后的旋转矩阵: \n{R_justify}")
	# 构造相机到标定板坐标系的空间变换    
	T_cam2board2 = np.eye(4)
	# 使用矫正后的姿态
	T_cam2board2[:3, :3] = R_justify
	# 使用矫正后的位置
	T_cam2board2[:3, 3] = t_cam2board_new
	return T_cam2board2

def o3d_mesh_convex(omesh, color=[0.8, 0.5, 0.5]):
	'''计算Open3D的凸包'''
	# 计算Mesh的凸包
	omesh_convex = omesh.compute_convex_hull()[0]
	omesh_convex.paint_uniform_color(color)
	omesh_convex.compute_vertex_normals()
	return omesh_convex

def create_o3d_box(width, height, depth, color=[0, 0.5, 0.5]):
	'''创建Open3D格式的Box'''
	o3d_box = o3d.geometry.TriangleMesh.create_box(width, height, depth)
	o3d_box.compute_vertex_normals()
	o3d_box.paint_uniform_color(color)
	# 注意事项，Open3D中box默认坐标系是在左下角的。 并不在中心位置
	# 所以这里将Box转换为跟fcl中的box一样，中心点定义在正中心
	o3d_box.transform(Transform.dmat(-0.5*width, -0.5*height, -0.5*depth))
	return o3d_box
