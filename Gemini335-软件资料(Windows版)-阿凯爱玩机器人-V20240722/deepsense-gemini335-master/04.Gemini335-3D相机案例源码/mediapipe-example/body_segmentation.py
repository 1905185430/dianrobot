'''
人体图像分割
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

# 人体分割模型初始化
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
		model_selection=1) 

cv2.namedWindow('Body Segmentation', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
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
	results = selfie_segmentation.process(img_rgb)
	# 获取人的Mask
	human_mask = np.uint8(results.segmentation_mask) * 255
	
	# 数学形态学运算 
	# 创建 核
	kernel = np.ones((13,13), np.uint8)
	# 膨胀
	human_mask = cv2.dilate(human_mask, kernel, iterations=1)
	
	# 创建纯色背景图片
	# 注: 你也可以替换为其他的相同尺寸的背景图
	bg_image = np.zeros(img_bgr.shape, dtype=np.uint8)
	bg_image[:] = [125, 125, 125]
	# 抠图
	human_mask_ch3 = cv2.cvtColor(human_mask, cv2.COLOR_GRAY2BGR)
	canvas = np.where(human_mask_ch3, img_bgr, bg_image)
	
	# 显示图像
	cv2.imshow('Body Segmentation', canvas)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break

camera.release()
