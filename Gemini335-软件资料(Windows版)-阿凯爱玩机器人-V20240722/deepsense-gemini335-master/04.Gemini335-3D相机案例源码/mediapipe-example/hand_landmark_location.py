'''
手掌关键点检测与定位
- 获取手掌0号点在相机坐标系下的三维坐标
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
# 获取彩图
color_img = camera.read_color_img()
# 获取深度图
depth_img = camera.read_depth_img()
# 生成画布
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
	if results.multi_hand_landmarks is not None:
		pixel_landmark = results.multi_hand_landmarks[0].landmark
		# 获取Landmark的像素坐标
		# 获取第一个关键点的像素坐标
		mark_idx = 0
		px = int(pixel_landmark[mark_idx].x * camera.img_width)
		py = int(pixel_landmark[mark_idx].y * camera.img_height)

		dp_h, dp_w = depth_img.shape
		#根据像素坐标获取深度值
		if px < dp_w and py < dp_h:
			# 获取0点的x、y坐标
			depth_value = depth_img[py, px]
			# 深度值无效，查询相邻的区域取均值
			if depth_value==0:
				roi = depth_img[py-10:py+10, px-10:px+10]
				index = np.where(roi !=  0)
				if len(index) != 0:
					depth_mean = np.mean(roi[index])
					print(f"深度值无效 depth_value={depth_value} 均值 {depth_mean}")
					if not np.isnan(depth_mean) and depth_mean != 0:
						depth_value = depth_mean
			if depth_value != 0.0:
				cam_point3d = camera.depth_pixel2cam_point3d(\
											px, py, depth_value=depth_value)#由像素坐标转化为下相机坐标
				cam_x, cam_y, cam_z = cam_point3d
				print(f"彩色相机坐标系下的坐标: [{cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}], 单位mm")			
		
	cv2.imshow('Hands', canvas)
	if cv2.waitKey(5) & 0xFF == 27:
		break
camera.release()
