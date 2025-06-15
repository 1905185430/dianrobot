'''
人脸检测-MediaPipe
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import numpy as np
import cv2
import mediapipe as mp

class FaceDetector:
	CONFIDENCE_THRESHOLD = 0.5 # 置信度阈值
	def __init__(self, camera):
		self.camera = camera
		# 创建人脸检测器
		mp_face_detection = mp.solutions.face_detection
		self.face_detection = mp_face_detection.FaceDetection(
				model_selection=0, \
				min_detection_confidence=self.CONFIDENCE_THRESHOLD)
	
	def get_roi(self, detection):
		'''从detection对象中提取ROI'''
		# 获取包围框
		bbox = detection.location_data.relative_bounding_box
		# 将百分比转换为像素
		x = int(bbox.xmin * self.camera.img_width)
		y = int(bbox.ymin * self.camera.img_height)
		w = int(bbox.width * self.camera.img_width)
		h = int(bbox.height * self.camera.img_height)
		return [x, y, w, h]
	
	def detect_face(self, img_bgr, canvas=None):
		'''人脸检测'''
		# 返回结果
		rect_list = [] # 矩形区域列表
		conf_list = [] # 置信度列表
		
		# 颜色空间转换 BGR色彩空间转换为RGB色彩空间
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		# 为了提高性能， 将图像标记为只读模式
		img_rgb.flags.writeable = False
		# 人脸检测
		results = self.face_detection.process(img_rgb)
		
		# 画布初始化
		if canvas is None:
			canvas = np.copy(img_bgr)
		if results.detections is None:
			# 没有有效人脸
			return rect_list, conf_list, canvas
		 
		for detection in results.detections:
			# 提取ROI
			rect = self.get_roi(detection)
			# 提取置信度
			conf = detection.score[0]
			# 添加到列表中
			rect_list.append(rect)
			conf_list.append(conf)
			# 绘制矩形框
			[x, y, w, h] = rect
			cv2.rectangle(canvas, (x, y), (x+w, y+h), (0,0,255), thickness=3)
		return rect_list, conf_list, canvas


if __name__ == "__main__":
	# 阿凯机器人工具箱
	from kyle_robot_toolbox.camera import Gemini335
	# 创建相机对象
	camera = Gemini335()
	# 创建人脸检测器
	detector = FaceDetector(camera)

	while True:
		# 采集图像
		img_bgr = camera.read_color_img()
		# 人脸检测
		rect_list, conf_list, canvas = detector.detect_face(img_bgr)
		# 为了有照镜子的感觉， 水平镜像一下图像
		cv2.imshow('Face Detection', cv2.flip(canvas, 1))
		if cv2.waitKey(5) & 0xFF == 27:
			break
	# 释放capture
	cap.release()

