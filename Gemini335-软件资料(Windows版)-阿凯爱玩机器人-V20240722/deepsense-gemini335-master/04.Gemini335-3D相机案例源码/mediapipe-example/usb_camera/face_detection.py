'''
人脸检测-MediaPipe
'''
import numpy as np
import cv2
import mediapipe as mp

# 创建USB相机
cap = cv2.VideoCapture(0)
# 创建人脸检测器
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
		model_selection=0, \
		min_detection_confidence=0.7)
# 创建可视化工具
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
	ret, img_bgr = cap.read()
	if not ret:
	  continue
	# 颜色空间转换 BGR色彩空间转换为RGB色彩空间
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	# 为了提高性能， 将图像标记为只读模式
	img_rgb.flags.writeable = False
	# 人脸检测
	results = face_detection.process(img_rgb)
	# 创建模式
	canvas = np.copy(img_bgr)
	if results.detections is not None:
		# 编译所有的结果
		for detection in results.detections:
			mp_drawing.draw_detection(canvas, detection)
	# 为了有照镜子的感觉， 水平镜像一下图像
	cv2.imshow('MediaPipe Face Detection', cv2.flip(canvas, 1))
	if cv2.waitKey(5) & 0xFF == 27:
		break
# 释放capture
cap.release()

