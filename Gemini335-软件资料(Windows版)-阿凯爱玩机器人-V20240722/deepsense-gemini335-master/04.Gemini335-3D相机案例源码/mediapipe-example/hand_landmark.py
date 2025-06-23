'''
手掌关键点检测
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335

# 创建相机对象
camera = Gemini335()

# min_detection_confidence: 置信度阈值 
# min_tracking_confidence: 跟踪阈值
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
		model_complexity=0,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5)

# 绘图
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
	# 采集图像
	# 注: 前几帧图像质量不好，可以多采集几次  
	# 另外确保画面中有手
	img_bgr = camera.read_color_img()
	# 为了左右手被正确识别， 需要镜像一下
	img_bgr = cv2.flip(img_bgr, 1)
	# 为了提高性能， 将图像标记为只读模式
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	img_rgb.flags.writeable = False
	# 手掌关键点检测
	results = hands.process(img_rgb)
	# 创建画布
	canvas = np.copy(img_bgr)
	if results.multi_hand_landmarks is not None:
		# 绘制关键点
		for hand_landmarks in results.multi_hand_landmarks:
			mp_drawing.draw_landmarks(
				canvas,
				hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())
		# 绘制关键点序号
		for hand_landmarks in results.multi_hand_landmarks:    
			for i, mark in enumerate(hand_landmarks.landmark):
				px = int(mark.x * camera.img_width)
				py = int(mark.y * camera.img_height)
				cv2.putText(canvas, str(i), (px-25, py+5), \
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	
	cv2.imshow('Hands', canvas)
	if cv2.waitKey(5) & 0xFF == 27:
		break
camera.release()
