'''Gemini2 3D相机 Python SDK(pyorbbec版)
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import os
import sys
import pkg_resources
import time
import numpy as np
import cv2
import open3d as o3d
import logging
import yaml

# 获取logger实例
logger = logging.getLogger("Gemini335")
# 指定日志的最低输出级别
logger.setLevel(logging.WARN)

# 自定义库
from kyle_robot_toolbox.camera import Camera
from kyle_robot_toolbox.open3d import *

# 获取阿凯机器人工具箱的安装路径
dist = pkg_resources.get_distribution("kyle_robot_toolbox")
kyle_robot_toolbox_path = os.path.join(dist.location, 'kyle_robot_toolbox')
# 根据操作系统类型, 导入不同的pyorbbecsdk动态链接库
if os.name == 'nt':
	# Windows操作系统
	pyorbbecsdk_path = os.path.join(kyle_robot_toolbox_path, 'lib', 'pyorbbecsdk', 'windows')	
elif os.name == 'posix':
	# Ubuntu操作系统(Linux)
	pyorbbecsdk_path = os.path.join(kyle_robot_toolbox_path, 'lib', 'pyorbbecsdk', 'linux')
print("[INFO][pyorbbecsdk]添加动态链接库检索路径")
print(pyorbbecsdk_path)

sys.path.append(pyorbbecsdk_path)

# 奥比中光 pyorbbecsdk
from pyorbbecsdk import *


class Gemini335(Camera):
	'''Gemini335 3D相机类'''
	# 像素分辨率
	img_width = 1280
	img_height = 720
	# 距离阈值范围(单位mm)
	min_distance = 150
	max_distance = 10000
	# 距离过滤器 
	distance_filter_enable = False
	# 缓冲区尺寸
	buffer_size = 2
	# 对齐模式设置为软件对齐
	align_mode = "SW"
	# LDP开关
	# 近距离保护, 一般用不上, 反而会带来麻烦 
	ldp_enable = False

	def __init__(self, serial_num=None, yaml_path=None):
		# 注: capture_id弃用
		# 父类初始化
		super().__init__(img_width=self.img_width, \
				img_height=self.img_height)

		# 初始化设备
		self.device = None
		# 管道
		self.pipeline = None
		# 初始管道
		self.connect_device(serial_num=serial_num)
		# 初始化视频流
		if yaml_path is None:
			self.init_pipeline()
		else:
			self.load_config(yaml_path)
		# 载入相机标定参数
		self.load_cam_calib_data()
	
	def connect_device(self, serial_num=None):
		'''连接设备'''
		# 设置日志等级为ERROR 
		# 这样不会频繁的打印日志信息
		contex = Context()
		contex.set_logger_level(OBLogLevel.ERROR)

		# 查询设备列表 
		device_list = contex.query_devices()
		# 获取设备个数
		device_num = device_list.get_count()

		
		if device_num == 0:
			logger.error("[ERROR]没有设备连接")
			return False, None
		else:
			# 没有指定序列号
			if serial_num is None:
				logger.info(f"[INFO] 检测到{device_num}个设备")
				# 获取特定索引下的设备序列号
				serial_num = device_list.get_device_serial_number_by_index(0)
				logger.info(f"[INFO]设备序列号为: {serial_num}")

			try:
				if self.device is not None:
					logger.info("[INFO]设备已经创建过, 在创建前先释放设备。")
					# 断开设备, 重新创建连接
					try:
						# 设备断开连接
						self.device.reboot()
						# 删除设备
						del self.device
					except Exception as e:
						logger.error("[ERROR]在断开设备的时候，出现报错 ")
						print(e)
						# 删除设备
						del self.device
				logger.info("[INFO]重新刷新设备列表")
				# 预留3s, 给设备重新连接预留时间
				for i in range(30):
					# 重新查询设备列表 
					device_list = contex.query_devices()
					# 检查是否有设备接入
					if device_list.get_count() != 0:
						# 检测到设备接入, 就退出循环
						break
					time.sleep(0.1)
				
				if device_list.get_count() != 0:
					# 根据设备序列号创建设备
					self.device = device_list.get_device_by_serial_number(serial_num)
					logger.info("[INFO]设备成功创建连接")
					return True, self.device
				else:
					logger.error("[ERROR] 没有检测到设备连接")
					return False, None
			except OBError as e:
				logger.error("[ERROR] 设备连接失败, 检查是不是有其他脚本/上位机软件占用了相机设备")
				logger.error("需要将其他脚本/上位机都关掉之后， 重新当前脚本并重试")
				logger.error("当然也有可能是在当前的脚本中，相机设备已经创建了连接。 因此在重新连接前，先释放设备。")
				logger.error("详细信息: ")
				logger.error(e)
				return False, None
	
	def init_pipeline(self):
		'''初始化管道'''
		# 将device传入Pipeline
		self.pipeline = Pipeline(self.device)
		# 创建配置信息对象
		self.config = Config()

		# 获取彩图选项列表
		color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)   
		# 获取深度图选项列表
		depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)   

		# 手动创建彩色视频流配置信息
		width = 1280 # 图像宽度
		height = 720 # 图像高度
		fmt = OBFormat.MJPG # 图像格式
		fps = 30 # 帧率
		color_profile = color_profile_list.get_video_stream_profile(width, height, fmt, fps)
		# 在配置信息里面定义彩色视频流的基本信息
		self.config.enable_stream(color_profile)
		
		# 手动创建深度图视频流配置信息
		width = 1280 # 图像宽度
		height = 720 # 图像高度
		fmt = OBFormat.Y16 # 图像格式
		fps = 30 # 帧率
		depth_profile = depth_profile_list.get_video_stream_profile(width, height, fmt, fps)
		# 在配置信息里面定义深度图视频流的基本信息
		self.config.enable_stream(depth_profile)

		# 注: Gemini335不支持硬件对齐 
		# 选择软件对齐
		self.config.set_align_mode(OBAlignMode.SW_MODE)
		# 帧同步 ？ 
		# pipeline.enable_frame_sync()
		# 禁用LDP
		self.device.set_bool_property(OBPropertyID.OB_PROP_LDP_BOOL, self.ldp_enable)
		# 开启并配置管道
		self.pipeline.start(self.config)
		
		print("[INFO] 相机初始化过程中，前几帧可能会捕获异常的警告(WARN)，是正常现象。")
		# 初始化多采集一些图像
		self.empty_cache(frame_num=10)
		print("[INFO] 相机初始化完成，缓冲区已清理")
		return self.pipeline

	def color_frame_to_bgr_img(self, frame):
		'''将彩图数据帧转换为numpy格式的BGR彩图'''
		width = frame.get_width()
		height = frame.get_height()
		color_format = frame.get_format()
		data = np.asanyarray(frame.get_data())
		image = np.zeros((height, width, 3), dtype=np.uint8)
		if color_format == OBFormat.RGB:
			image = np.resize(data, (height, width, 3))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		elif color_format == OBFormat.MJPG:
			image = cv2.imdecode(data, cv2.IMREAD_COLOR)
		else:
			logger.error("[ERROR] 不支持彩图数据格式: {}".format(color_format))
			return None
		return image

	def depth_frame_to_depth_img(self, frame):
		'''深度数据帧转换为深度图'''
		# 获取深度图宽度
		width = frame.get_width()
		# 获取深度图高度
		height = frame.get_height()
		# 获取深度图尺度
		scale = frame.get_depth_scale()
		# 转换为numpy的float32格式的矩阵
		depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
		depth_data = depth_data.reshape((height, width))
		depth_data = depth_data.astype(np.float32) * scale
		return depth_data

	def get_color_image_from_frames(self, frames):
		'''在Frames中提取彩图'''
		if frames is None:
			return None
		# 获取彩图数据帧
		color_frame = frames.get_color_frame()
		if color_frame is None:
			return None
		return self.color_frame_to_bgr_img(color_frame)

	def get_depth_image_from_frames(self, frames):
		'''在Frames中提取深度图'''
		if frames is None:
			return None
		# 获取深度图数据帧
		depth_frame = frames.get_depth_frame()
		if depth_frame is None:
			return None
		return self.depth_frame_to_depth_img(depth_frame, \
				self.min_distance, self.max_distance)
	
	def read_color_img(self, retry_num=10, timeout_ms=3500):
		'''拍照, 只获取彩图
		@retry_num: 重试次数
		@timeout_ms: 超时等待时间, 单位ms
		'''
		# 彩图
		color_img = None

		# 重复N次
		for i in range(retry_num):
			# 获取数据帧
			frames = self.pipeline.wait_for_frames(timeout_ms)

			if frames is None:
				logger.warn("[WARN] 数据帧获取失败, 请重试")
				continue
			else:
				logger.info("[INFO] 数据帧读取成功")

				# 从数据帧frames中获取彩图数据帧
				color_frame = frames.get_color_frame()
				if color_frame is None:
					logger.warn("[WARN] 彩图获取失败")
					continue
				else:
					# 转换为OpenCV格式的彩图数据格式
					color_img = self.color_frame_to_bgr_img(color_frame)
					if color_img is None:
						logger.warn("[WARN] 彩图数据解析失败")
						continue
					else:
						logger.info("[INFO] 彩图获取成功")
						return color_img
					
		return None

	def read_depth_img(self, retry_num=10, timeout_ms=3500):
		'''拍照, 只采集深度图
		@retry_num: 重试次数
		@timeout_ms: 超时等待时间, 单位ms
		'''
		# 深度图
		depth_img = None

		# 重复10次
		for i in range(retry_num):
			# 获取数据帧
			frames = self.pipeline.wait_for_frames(timeout_ms)

			if frames is None:
				logger.warn("[WARN] 数据帧获取失败, 请重试")
				continue
			else:
				logger.info("[INFO] 数据帧读取成功")
				
				# 从数据帧frames中获取深度图数据帧
				depth_frame = frames.get_depth_frame()
				if depth_frame is None:
					logger.warn("[WARN] 深度图获取失败")
				else:
					# 转换为OpenCV格式的彩图数据格式
					depth_img = self.depth_frame_to_depth_img(depth_frame)
					if depth_img is None:
						logger.warn("[WARN] 深度图数据解析失败")
						continue
					else:
						logger.info("[INFO] 深度图获取成功")
						return depth_img
		return None

	def read(self, retry_num=10, timeout_ms=3500):
		'''拍照, 同时采集彩图与深度图
		@pipeline: 数据管道
		@retry_num: 重试次数
		@timeout_ms: 超时等待时间, 单位ms
		'''
		# 彩图
		color_img = None
		# 深度图
		depth_img = None

		# 重复10次
		for i in range(retry_num):
			# 获取数据帧
			frames = self.pipeline.wait_for_frames(timeout_ms)

			if frames is None:
				logger.warn("[WARN] 数据帧获取失败, 请重试")
				continue
			else:
				logger.info("[INFO] 数据帧读取成功")

				# 从数据帧frames中获取彩图数据帧
				color_frame = frames.get_color_frame()
				if color_frame is None:
					logger.warn("[WARN] 彩图获取失败")
					continue
				else:
					# 转换为OpenCV格式的彩图数据格式
					color_img = self.color_frame_to_bgr_img(color_frame)
					if color_img is None:
						logger.warn("[WARN] 彩图数据解析失败")
						continue
					else:
						logger.info("[INFO] 彩图获取成功")
				
				# 从数据帧frames中获取深度图数据帧
				depth_frame = frames.get_depth_frame()
				if depth_frame is None:
					logger.warn("[WARN] 深度图获取失败")
				else:
					# 转换为OpenCV格式的彩图数据格式
					depth_img = self.depth_frame_to_depth_img(depth_frame)
					if depth_img is None:
						logger.warn("[WARN] 深度图数据解析失败")
						continue
					else:
						logger.info("[INFO] 深度图获取成功")
						return color_img, depth_img
		return None, None

	def get_intrinsic(self):
		'''获取RGB相机内参'''
		# 获取相机参数
		self.camera_param = self.pipeline.get_camera_param()
		# 获取彩色相机内参
		fx = self.camera_param.rgb_intrinsic.fx
		fy = self.camera_param.rgb_intrinsic.fy
		cx = self.camera_param.rgb_intrinsic.cx
		cy = self.camera_param.rgb_intrinsic.cy
		# 创建内参矩阵
		intrinsic = np.identity(3)
		intrinsic[0, 0] = fx
		intrinsic[1, 1] = fy
		intrinsic[0, 2] = cx
		intrinsic[1, 2] = cy
		return intrinsic
	
	def get_distortion(self):
		'''获取RGB相机的畸变系数'''
		# 获取相机参数
		self.camera_param = self.pipeline.get_camera_param()
		d = self.camera_param.rgb_distortion
		# 畸变系数列表顺序 [k1, k2, p1, p2, k3, k4, k5, k6]
		distortion = [d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6]
		distortion = np.float64(distortion)
		return distortion
	
	def load_cam_calib_data(self, load_distortion=False):
		'''设置相机内参'''
		# 载入相机内参
		self.intrinsic = self.get_intrinsic()
		if not load_distortion:
			self.set_parameter(self.intrinsic)
		else:
			# 载入相机畸变系数
			self.distortion = self.get_distortion()
			# 畸变矫正相关
			# 根据图像缩放尺寸， 重新计算相机内参矩阵 
			self.intrinsic_new, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic, self.distortion,\
				(self.img_width, self.img_height), 0, (self.img_width, self.img_height))
			self.remap_x, self.remap_y = cv2.initUndistortRectifyMap(self.intrinsic, self.distortion, None,\
				self.intrinsic_new, (self.img_width, self.img_height), 5)
			# 根据相机标定参数
			# 提取图像中心(cx, cy)与焦距f(单位：像素)
			self.fx = self.intrinsic_new[0, 0]
			self.fy = self.intrinsic_new[1, 1]
			self.f = (self.fx + self.fy) / 2
			# 图像中心的坐标
			self.cx = self.intrinsic_new[0, 2]
			self.cy = self.intrinsic_new[1, 2]
			# 生成视场角等相关参数
			self.alpha1 = np.arctan(self.cy/self.f)
			self.alpha2 = np.arctan((self.img_height-self.cy)/self.f)
			self.beta1 = np.arctan(self.cx/self.f)
			self.beta2 = np.arctan((self.img_width-self.cx)/self.f)
			# 创建Open3D的内参矩阵对象
			self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
				self.img_width, self.img_height, \
				self.fx, self.fy, self.cx, self.cy)
	
	def empty_cache(self, frame_num=5):
		'''清空摄像头的缓冲区'''
		for i in range(frame_num):
			self.read()
	
	def release(self):
		'''释放相机'''
		# 停止Pipeline
		self.pipeline.stop()
		# 关闭所有视频流
		self.config.disable_all_stream()
		
	def depth_img2canvas(self, depth_img, min_distance = None,\
			max_distance = None):
		'''将深度图转换为画布
		注意事项: 在使用的时候，手动传入实际的工作距离范围，可以更好的进行
		可视化
		'''
		# 判断是否全部无效
		valid_point_index = depth_img != 0
		if len(valid_point_index) == 0:
			return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
		# 根据实际情况调整最大距离与最远距离
		valid_depth = depth_img[valid_point_index]
		# if len(valid_depth) == 0:
		if len(valid_depth) < 100:
			# 有效深度太少 返回空
			return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

		if min_distance is None:
			min_distance = np.min(valid_depth)-150
		if max_distance is None:
			max_distance = np.max(valid_depth)+150		
		# 距离阈值的单位是mm
		depth_img_cut = np.copy(depth_img).astype(np.float64)
		depth_img_cut[depth_img < min_distance] = min_distance
		depth_img_cut[depth_img > max_distance] = max_distance
		# 归一化
		depth_img_norm = np.uint8(255.0*(depth_img_cut-min_distance)/(max_distance - min_distance))
		# 转换为
		depth_colormap = cv2.applyColorMap(depth_img_norm, cv2.COLORMAP_JET)
		return depth_colormap
	
	def get_pcd(self, color_image, depth_image,\
			mask=None, min_distance=None, max_distance=None,\
			camera="rgb_camera"):
		'''从彩图与对齐后的深度图构建Open3D格式的点云数据'''
		# 对深度图进行过滤
		# 超出这个范围内的都当作为无效距离
		depth_image_cut = np.copy(depth_image)
		# 根据Mask来筛选深度像素(点云)
		if mask is not None:
			mask_pixel_idx = (mask == 0)
			depth_image_cut[mask_pixel_idx] = 0
		# 获取RGB相机下的点云
		return super().get_pcd(color_image, depth_image_cut, \
			min_distance=min_distance, max_distance=max_distance)
	

	def get_pcd2(self, depth_image,\
			mask=None, min_distance=None, max_distance=None, \
	   		rgb_color=None, camera="rgb_camera"):
		'''从深度图构建Open3D格式的点云数据
  		'''
		# 构造彩图
		if rgb_color is None:
			# 给点云赋值为可视化深度图的颜色
			color_image = self.depth_img2canvas(depth_image, min_distance=min_distance, \
				max_distance=max_distance)
		else:
			# 给点云赋值为同一个颜色(rgb_color)
			img_h, img_w = depth_image.shape
			color_image = np.ones((img_h, img_w, 3), dtype=np.uint8)
			color_image[:, :] = rgb_color
		# 对深度图进行过滤
		# 超出这个范围内的都当作为无效距离
		depth_image_cut = np.copy(depth_image)
		# 根据Mask来筛选深度像素(点云)
		if mask is not None:
			mask_pixel_idx = (mask == 0)
			depth_image_cut[mask_pixel_idx] = 0
		# 开启距离过滤器
		if self.distance_filter_enable:
			if min_distance is None:
				min_distance = self.min_distance
			depth_image_cut[depth_image < min_distance] = 0
			if max_distance is None:
				max_distance = self.max_distance
			depth_image_cut[depth_image > max_distance] = 0
		return self.get_pcd(color_image, depth_image_cut, \
			min_distance=min_distance,\
			max_distance=max_distance, \
			camera=camera)
		
	def geometry_camera(self, T_world2cam=np.eye(4)):
		'''获取相机的可视化模型'''
		return geometry_camera(self.intrinsic, T_world2cam, \
			 self.img_width, self.img_height)
	
	def load_config(self, yaml_path):
		'''载入配置文件'''
		# 加载配置文件
		# 载入Gemini2相机的配置文件
		with open(yaml_path, 'r', encoding='utf-8') as f:
			camera_config_yaml = yaml.load(f.read(), Loader=yaml.SafeLoader)
		# 停止视频流(如果存在)
		if self.pipeline is not None:
			self.pipeline.stop()
		# 获取彩色相机配置
		rgb_config = camera_config_yaml['rgb_camera']
		logger.info("[彩色相机配置]")

		# 配置曝光与相关参数
		if rgb_config['auto_exposure']:
			# 自动曝光
			logger.info(f"- 打开自动曝光")
			self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
			# 设置亮度
			logger.info(f"- 亮度: {rgb_config['brightness']}")
			# 设置相机亮度
			self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, rgb_config['brightness'])
		else:
			# 关闭自动曝光
			logger.info(f"- 关闭自动曝光")
			self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
			# 设置曝光值
			logger.info(f"- 曝光: {rgb_config['exposure']}")
			self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, rgb_config['exposure'])
			# 设置增益
			logger.info(f"- 增益: {rgb_config['gain']}")
			self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, rgb_config['gain'])

		# 配置白平衡
		if rgb_config['auto_white_balance']:
			# 打开自动白平衡
			logger.info(f"- 打开自动白平衡")
			self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
		else:
			# 关闭自动白平衡
			logger.info(f"- 关闭自动白平衡")
			self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, False)
			# 设置白平衡
			logger.info(f"- 白平衡: {rgb_config['white_balance']}")
			self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, rgb_config['white_balance'])

		# 其他配置项
		# 设置相机锐度
		logger.info(f"- 锐度: {rgb_config['sharpness']}")
		self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, rgb_config['sharpness'])
		# 设置饱和度
		logger.info(f"- 饱和度: {rgb_config['saturation']}")
		self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_SATURATION_INT, rgb_config['saturation'])
		# 设置对比度
		logger.info(f"- 对比度: {rgb_config['contrast']}")
		self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, rgb_config['contrast'])
		# 电力线频率
		logger.info(f"- 电力线频率: {rgb_config['power_line_frequency']}")
		self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, rgb_config['power_line_frequency'])
		
		# 获取深度相机的配置
		depth_config = camera_config_yaml['depth_camera']
		logger.info("[深度相机配置]")

		# 配置曝光与相关参数
		if depth_config['auto_exposure']:
			logger.info(f"- 打开自动白平衡")
			self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, True)
		else:
			logger.info(f"- 关闭自动白平衡")
			self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, False)
			logger.info(f"- 曝光: {depth_config['exposure']}")
			self.device.set_int_property(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, depth_config['exposure'])
			self.logger.info(f"- 增益: {depth_config['gain']}")
			self.device.set_int_property(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, depth_config['gain'])

		# 精度等级
		# logger.info(f"- 配置深度精度等级: {depth_config['precision_level']}")
		# self.device.set_int_property(OBPropertyID.OB_PROP_DEPTH_PRECISION_LEVEL_INT, depth_config['precision_level'])

		# 深度工作模式
		mode_list = self.device.get_depth_work_mode_list()
		logger.info(f"- 设置深度工作模式: {mode_list.get_name_by_index(depth_config['depth_work_mode'])}")
		mode = mode_list.get_depth_work_mode_by_index(depth_config['depth_work_mode'])
		self.device.set_depth_work_mode(mode)

		# 初始化视频流
		self.init_pipeline()

