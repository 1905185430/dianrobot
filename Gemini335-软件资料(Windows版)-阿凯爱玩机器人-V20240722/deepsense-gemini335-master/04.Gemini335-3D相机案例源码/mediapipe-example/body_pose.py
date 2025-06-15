'''
人体关键点检测
注: 第一次执行的时候， 使用sudo权限
因为要下载一个模型文件。 
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

# 创建人体关键点检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5)

# 创建绘图工具
mp_drawing = mp.solutions.drawing_utils
# 创建绘图风格
mp_drawing_styles = mp.solutions.drawing_styles

cv2.namedWindow('Body Pose', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

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
	# 人体关键点检测
	results = pose.process(img_rgb)
	# 创建画布
	canvas = np.copy(img_bgr)

	if results.pose_landmarks is None:
		# print("没有检测到人体")
		pass
	else:
		mp_drawing.draw_landmarks(
			canvas,
			results.pose_landmarks,
			mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=\
				mp_drawing_styles.get_default_pose_landmarks_style())
	# 显示图像
	cv2.imshow('Body Pose', canvas)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break

camera.release()