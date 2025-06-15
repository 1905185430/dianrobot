'''
手掌关键点检测-使用Open3D来做可视化
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import json
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335
from kyle_robot_toolbox.open3d import *
# 自定义库
# - 手掌可视化库
from hand_visualizer import HandVisualizer

# 创建相机对象
camera = Gemini335()

# 创建可视化窗口
visualizer = HandVisualizer()
visualizer.create_window()

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

# 配置视角
json_path = "config/hand/render_option.json"
trajectory = json.load(open(json_path, "r", encoding="utf-8"))
view_point = trajectory["trajectory"][0]
def set_view_control():
	'''控制视野'''
	global visualizer
	global view_point
	ctr = visualizer.visualizer.get_view_control()
	ctr.set_front(np.float64(view_point["front"]))
	ctr.set_lookat(np.float64(view_point["lookat"]))
	ctr.set_up(np.float64(view_point["up"]))
	ctr.set_zoom(np.float64(view_point["zoom"]))

cv2.namedWindow('Hands', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

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

		
		# 获取关键点在手掌等效中心坐标系下的三维坐标 [x, y, z]
		# 单位m
		world_landmark = results.multi_hand_world_landmarks[0].landmark
		# 获取三维点集
		point3d_list = []
		for pidx, point in enumerate(world_landmark):
			# 追加3D列表
			point3d_list.append([point.x, point.y, point.z])
		# 关键点类型转换
		point3d_list = np.float64(point3d_list)
		# 3D窗口更新
		visualizer.update_hand(point3d_list)
	else:
		visualizer.reset_hand()
	# 控制视野
	set_view_control()
	# 可视化窗口更新
	visualizer.step()

	cv2.imshow('Hands', canvas)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
visualizer.destroy_window()