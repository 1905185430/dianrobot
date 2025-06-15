'''
人脸跟踪
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import time
import math
import cv2
import numpy as np
# 人脸检测
from face_detection import FaceDetector


class FaceTracker:
	'''人脸跟踪器'''
	RESAMPLE_IMG_COUNT = 10			# 重采样频率  多少fps重新采一次
	NEIGHBOR_ROI_MAX_DISTANCE = 100 # 像素距离阈值
	ENABLE_TRACKER = True 			# 追踪器是否开启
	def __init__(self, camera):
		self.camera = camera
		# 图像计数
		self.img_cnt = 0
		# 创建检测器
		self.detector = FaceDetector(camera)
		# 创建跟踪器
		if self.ENABLE_TRACKER:
			self.tracker = cv2.TrackerKCF_create()
			self.is_tracker_init = False
			self.track_roi_confidence = 1.0 # 跟踪框的置信度
		# 上次检测到人脸的中心位置
		self.last_roi = None 
	def rect2center(self, rect):
		'''ROI矩形框->中心'''
		x, y, w, h = rect
		return int(x+w/2), int(y+h/2)

	def rect_distance(self, rect1, rect2):
		'''计算矩形框之间的距离'''
		x1, y1 = self.rect2center(rect1)
		x2, y2 = self.rect2center(rect2)
		return math.sqrt((x1-x2)**2 + (y1-y2)**2)

	@property
	def last_center(self):
		if self.last_roi is None:
			return None
		return self.rect2center(self.last_roi)
	
	def is_old_roi_exist(self, rects, confidence_list):
		'''判断旧的ROI是否还在'''
		min_distance = 1000000.0
		neighbor_roi = None
		# 检查最近的ROI是否还在
		if self.last_roi is None or len(rects) == 0:
			return False, None
		x, y, w, h = self.last_roi
		last_cx = x + w / 2
		last_cy = y + h / 2
		# 如果存在则计算与之前ROI中心距离的最小值
		rect_num = len(rects)
		for i in range(rect_num):
			conf = confidence_list[i]
			rect = rects[i]
			x, y, w, h = rect
			cur_cx = x + w / 2
			cur_cy = y + h / 2
			cur_distance = math.sqrt((cur_cx - last_cx)**2 + (cur_cy - last_cy)**2)
			if cur_distance < min_distance:
				min_distance = cur_distance
				neighbor_roi = rect
		if neighbor_roi is not None and min_distance <= self.NEIGHBOR_ROI_MAX_DISTANCE:
			# print(f"neighbor_roi : {neighbor_roi} min_dis : {min_distance}")
			return True, neighbor_roi
		return False, None
	
	def select_target_rect(self, img, rects, confidence_list):
		'''选择目标矩形区域'''
		# 条件选择为置信度最高的
		if len(rects) == 0:
			return None
		# 判断原来跟踪的对象还在不在
		if self.last_roi is not None:
			is_valid, rect = self.is_old_roi_exist(rects, confidence_list)
			if is_valid:
				# 原来的对象还在，不需要重新选择
				return rect
		

		if self.last_roi is None:
			# 选择置信度最高的
			return rects[np.argmax(confidence_list)]
		else:
			# 选择一次距离上次检测的RECT距离最近的
			return min(rects, key=lambda rect:self.rect_distance(rect, self.last_roi))

	def update(self, img):
		'''更新跟踪器的状态'''
		age_idx_list = None
		# 图像拷贝
		self.canvas = np.copy(img)
		# 图像计数自增
		self.img_cnt += 1
		is_success = False
		# 上一帧有参考 跟踪器更新
		if self.is_tracker_init:
			is_success, box = self.tracker.update(img)
			if is_success:
				# 跟踪器检测成功, 提取roi区域
				last_roi = [int(v) for v in box]
				x, y, w, h = last_roi
				# 判断跟踪器里面的ROI是不是人脸
				img_roi = img[y:y+h, x:x+w]
				rect_list, confidence_list, _ = self.detector.detect_face(img_roi)
				if len(confidence_list) == 0:
					self.track_roi_confidence -= 0.1
					if self.track_roi_confidence < 0:
						self.track_roi_confidence = 0
				else:
					self.track_roi_confidence = 1.0
				if self.track_roi_confidence <= 0:
					print("人脸跟踪丢失")
					is_success = False
					self.is_tracker_init = False
				else:
					# print(f"跟踪器:  self.last_roi :  {self.last_roi}")
					self.last_roi = last_roi
			else:
				# 视觉跟踪失败
				self.is_tracker_init = False
		if not is_success or  self.last_roi is None or  ((not self.ENABLE_TRACKER or self.img_cnt >= self.RESAMPLE_IMG_COUNT) and len(rect_list) > 0):
			# 检测画面中的人脸
			rect_list, confidence_list, self.canvas = self.detector.detect_face(img_bgr)
			if len(confidence_list) == 0:
				self.last_roi = None
				self.is_tracker_init = False
			elif len(confidence_list) > 0:
				# print("confidect_max:  {}".format(np.max(np.float32(confidence_list))))
				# 选择最佳ROI
				last_roi = self.select_target_rect(img, rect_list, confidence_list)
				if self.ENABLE_TRACKER and last_roi is not None:
					# 识别到了矩形区域, 初始化tracker
					self.tracker = cv2.TrackerKCF_create()
					self.tracker.init(img, last_roi)
					self.is_tracker_init = True
					self.track_roi_confidence = 1.0
				if last_roi is not None:
					self.last_roi = last_roi
			
		# 在画面中绘制跟踪对象
		if self.last_roi is not None:
			x, y, w, h = self.last_roi
			cv2.rectangle(self.canvas, (x, y), (x + w, y + h), (0, 255, 255), 4)
			# 绘制颜色标签
			cv2.putText(self.canvas, text="face",\
				org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
				fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))

		return True

if __name__ == "__main__":
	# 阿凯机器人工具箱
	from kyle_robot_toolbox.camera import Gemini335
	# 创建相机对象
	camera = Gemini335()
	# 创建窗口
	cv2.namedWindow('Face Track', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

	# 创建跟踪器
	tracker = FaceTracker(camera)
	while True:
		t_start = time.time()
		# 采集图像
		img_bgr = camera.read_color_img()
		# 跟踪器更新
		tracker.update(img_bgr)
		# 绘制帧率
		fps = int(1/(time.time() - t_start))
		cv2.putText(tracker.canvas, text=f"FPS:{fps}",\
			org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
			fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
		cv2.imshow("Face Track", tracker.canvas)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
	cv2.destroyAllWindows()
	camera.release()