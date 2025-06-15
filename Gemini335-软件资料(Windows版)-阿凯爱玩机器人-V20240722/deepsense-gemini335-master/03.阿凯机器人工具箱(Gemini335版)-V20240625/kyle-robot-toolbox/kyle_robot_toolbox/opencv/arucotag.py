'''
ArucoTag检测与位姿估计
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import time
import yaml
import numpy as np
import cv2
from cv2 import aruco

# 阿凯机器人工具箱
# - 计算几何 直线2D
from kyle_robot_toolbox.geometry.line2d import line_cross_pt2

class ArucoTag:
	def __init__(self, cam, config_path='config/arucotag.yaml'):
		# 载入配置文件
		with open(config_path, 'r', encoding='utf-8') as f:
			self.config = yaml.load(f.read(), Loader=yaml.SafeLoader)
		# 赋值相机对象
		self.cam = cam
		# 选择ArucoTag的字典
		version_big, version_small = [int(v) for v in cv2.__version__.split(".")[:2]]
		# 注: OpenCV 4.7修改了API, 选择是否使用新的API
		self.use_new_api = version_big > 4 or (version_big == 4 and version_small >= 7)
		if self.use_new_api:
			self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
			self.aruco_params = aruco.DetectorParameters()
			self.aruco_detector = aruco.ArucoDetector(\
							self.aruco_dict, self.aruco_params)
		else:
			self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
			self.aruco_params = aruco.DetectorParameters_create()
		# ArucoTag的尺寸
		self.marker_size = self.config["aruco_size"]
		# 初始化尺寸相关参数
		self.aruco_size_mm = self.config["aruco_size"]
		self.aruco_size_m = self.aruco_size_mm / 1000.0
		if "box_width" in self.config:
			self.box_width_mm = self.config["box_width"]
			self.box_width_m = self.box_width_mm / 1000.0
		if "box_depth" in self.config:
			self.box_depth_mm = self.config["box_depth"]
			self.box_depth_m = self.box_depth_mm / 1000.0
		
	def find_aruco(self, img, canvas=None):
		'''检测ArucoTag, 只获取ArucoTag在图像中的像素坐标'''
		# 创建画布
		if canvas is None:
			canvas = np.copy(img)
		# 转换为灰度图
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		if self.use_new_api:
			# 检测画面中的ArucoTag
			corners, aruco_ids, rejected_img_pts = self.aruco_detector.detectMarkers(gray)
		else:
			# 检测画面中的ArucoTag
			corners, aruco_ids, rejected_img_pts = aruco.detectMarkers(gray, \
				self.aruco_dict, parameters=self.aruco_params)
		
		if aruco_ids is None or len (aruco_ids) == 0:
			# 画面中没有检测到ArucoTag
			return False, canvas, [], [], []
		# 计算ArucoTag的中心
		aruco_centers = []
		for corner in corners:
			# 提取角点
			a, b, c, d = corner.reshape((-1, 2))
			# 计算交点
			ret, center = line_cross_pt2(a, c, b, d)
			if ret:
				aruco_centers.append(center)
			else:
				print("Error: arucoTag corner don't have cross point {}".format(corner.reshape((-1, 2))))
		# 可视化
		# 绘制Marker的边框与绘制编号
		canvas = aruco.drawDetectedMarkers(canvas, corners, aruco_ids,  (0,255,0))
		# 格式转换
		aruco_centers = np.int32(aruco_centers)
		corners = np.float64(corners)
		return True, canvas, aruco_ids.reshape(-1), aruco_centers, corners

	def aruco_pose_estimate(self, img, canvas=None):
		'''ArucoTag位姿估计'''
		# 检测画面中的ArucoTag
		ret, canvas, aruco_ids, aruco_centers, corners = \
      		self.find_aruco(img, canvas=canvas)
		aruco_num = len(aruco_ids)
		# 画面中没有ArucoTag
		if aruco_num == 0:
			return False, canvas, [], [], [], []
		# ArucoTag位姿估计
		# 返回 旋转向量列表与平移向量列表
		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(\
      		corners, self.marker_size, self.cam.intrinsic_new, \
            self.cam.distortion_new)
		# 将旋转向量转换为旋转矩阵
		rmat = []
		for i in range(aruco_num):
			rmat_i, _ = cv2.Rodrigues(rvec[i])
			rmat.append(rmat_i)
		# 构造变换矩阵
		# 相机坐标系到ArucoTag坐标系的坐标变换
		T_cam2aruco = []
		for i in range(aruco_num):
			tmat = np.eye(4)
			tmat[:3, :3] = rmat[i]
			tmat[:3, 3] = tvec[i].reshape(-1)
			T_cam2aruco.append(tmat)
		# 可视化, 绘制ArucoTag坐标系
		for i in range(aruco_num):
			if self.use_new_api:
				# OpenCV 4.7
				cv2.drawFrameAxes(canvas, self.cam.intrinsic_new, \
					self.cam.distortion_new, rvec[i], tvec[i], \
					self.marker_size)
			else:
				aruco.drawAxis(canvas, self.cam.intrinsic_new, \
					self.cam.distortion_new, rvec[i], tvec[i], \
					self.marker_size)
		# 格式转换
		T_cam2aruco = np.float64(T_cam2aruco)
		return True, canvas, aruco_ids, aruco_centers, corners, T_cam2aruco
