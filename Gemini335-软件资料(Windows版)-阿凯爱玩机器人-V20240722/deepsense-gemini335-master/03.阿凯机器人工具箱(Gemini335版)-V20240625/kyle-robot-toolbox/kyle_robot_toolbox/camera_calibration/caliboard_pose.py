'''
动态识别Radon标定板并对其进行位姿估计
Astra版, 忽略相机畸变
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np # 引入numpy 用于矩阵运算
import cv2 # 引入opencv库函数
import yaml
# 阿凯机器人工具箱
from kyle_robot_toolbox.transform import *
from kyle_robot_toolbox.open3d.o3d_utils import \
		get_panel_normal_vector,normal_vector_redirect, \
		get_rmat_from_z
class CaliboardPose:
	'''获取标定板的位姿'''
	def __init__(self, camera, calibration, 
			px_offset=0, py_offset=0):
		'''构造器'''
		self.camera = camera
		# 相机内参
		self.K = self.camera.intrinsic_new
		# 图像平移，调整相机内参
		self.K[0, 2] += px_offset
		self.K[1, 2] += py_offset
		# 赋值畸变系数
		self.distortion = self.camera.distortion_new
		
		self.calibration = calibration
		self.n_row = calibration.corner_row
		self.n_column = calibration.corner_column
		self.world_points = calibration.world_points
		self.ceil_size = calibration.config['caliboard']['ceil_size']

	def get_caliboard_pose(self, img, \
	 		draw_corner=False, 
	   		draw_pose_str=False, 
	  		return_contour_points_2d=False):
		'''获取标定板的位姿'''
		# 转换为灰度图
		if len(img.shape) == 3:
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# 创建画布
			canvas = np.copy(img)
		else:
			img_gray = img
			# 创建画布
			canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
		# 识别标定板角点
		ret, corners, meta = cv2.findChessboardCornersSBWithMeta(\
	  			img_gray, (self.n_row, self.n_column), \
				flags=cv2.CALIB_CB_ACCURACY|cv2.CALIB_CB_EXHAUSTIVE, meta=4)
		
		if not ret:
			# 没有检测到标定板
			if return_contour_points_2d:
				return False, None, None, None, canvas
			else:
				return False, None, None, canvas

		if draw_corner:
			# 绘制标定板角点
			canvas = cv2.drawChessboardCorners(canvas, \
				(self.n_row, self.n_column), corners[:self.n_row], patternWasFound=True)

		# 3D点 为角点在标定板坐标系下的位姿
		points_3d = np.float64(self.world_points)
		# 2D点 为角点在图像上的亚像素坐标
		points_2d = np.float64(corners)
		# 求解PnP问题
		# 注: 基于EPNP的位姿估计精度比较差，改用
  		# solvePnPRansac + SOLVEPNP_ITERATIVE
		# ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, \
		# 						self.camera.intrinsic, self.camera.distortion,\
		# 						flags=cv2.SOLVEPNP_EPNP)
		ret, rvec, tvec, *_ = cv2.solvePnPRansac(points_3d, points_2d, \
						self.K, \
						self.distortion, \
						flags=cv2.SOLVEPNP_ITERATIVE)
		# print(f"tvec: {tvec}")
		# print(f"rvec: {rvec}")
		# 计算T_board2cam
		pose_cam2board = Pose()
		pose_cam2board.set_position(*tvec.reshape(-1), unit="mm")
		pose_cam2board.set_rotation_vector(rvec.reshape(-1))
		T_cam2board = pose_cam2board.get_transform_matrix()
		
		# 坐标轴长度( 单位 mm)
		axis_length = self.ceil_size * max(self.n_row, self.n_column) / 2 * 1.2
		# 获取标定板的宽度与高度
		# 跟坐标轴相关的3D点
		axis_points_3d = np.array([
			[0.0, 0.0, 0.0], 
			[axis_length,0.0, 0.0],
			[0.0, axis_length, 0.0],
			[0.0, 0.0, axis_length], \
	   		], dtype="float32")
		axis_points_2d, jacobian = cv2.projectPoints(axis_points_3d, 
								rvec, tvec, self.K, self.distortion)
		if return_contour_points_2d:
			# 返回连通域点集
			board_width = self.calibration.config["caliboard"]["board_width"]
			board_height = self.calibration.config["caliboard"]["board_height"]
			w_d2 = board_width*0.5
			h_d2 = board_height*0.5

			contour_points_3d = np.array([
					[w_d2, h_d2, 0.0], 
					[w_d2, -h_d2, 0.0],
					[-w_d2, -h_d2, 0.0],
					[-w_d2, h_d2, 0.0] ], dtype="float32")
			contour_points_2d, jacobian = cv2.projectPoints(contour_points_3d, 
								rvec, tvec, self.K, self.distortion)
			# 类型转换
			contour_points_2d = np.int32(contour_points_2d)

		# 转换为像素坐标(整数)
		axis_points_2d = axis_points_2d.reshape((-1, 2)).astype('int32')
		center, axis_x, axis_y, axis_z = axis_points_2d[:4]
		# 绘制坐标轴
		axis_thickness = 4
		cv2.line(canvas, center, axis_x, (0, 0, 255), axis_thickness)
		cv2.line(canvas, center, axis_y, (0, 255, 0), axis_thickness)
		# cv2.line(canvas, center, axis_z, (255, 0, 0), axis_thickness)
		# 绘制位姿信息
		if draw_pose_str:
			x, y, z = pose_cam2board.get_position(unit="mm")
			roll, pitch, yaw = [np.degrees(theta) for theta in  pose_cam2board.get_euler_angle()]
			posi_str = "POSI(mm): [{:.1f},{:.1f},{:.1f}]".format(x, y, z)
			rpy_str = "RPY(deg): [{:.1f},{:.1f},{:.1f}]".format(roll, pitch, yaw)
			# 选择字体
			font = cv2.FONT_HERSHEY_SIMPLEX
			canvas[:65, :500] = [255, 255, 255]
			cv2.putText(canvas, text=posi_str, org=(20, 20), fontFace=font, \
				fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
			cv2.putText(canvas, text=rpy_str, org=(20, 50), fontFace=font, \
				fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
			
		if return_contour_points_2d:
			return True, pose_cam2board, center, contour_points_2d, canvas
		else:
			return True, pose_cam2board, center, canvas
