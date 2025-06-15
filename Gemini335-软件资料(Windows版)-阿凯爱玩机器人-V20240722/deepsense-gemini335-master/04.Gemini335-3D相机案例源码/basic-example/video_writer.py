#-*- coding: UTF-8 -*-
'''
功能描述：视频录制演示代码
----------------------------------------------
@作者: 阿凯爱玩机器人
@QQ: 244561792
@微信: xingshunkai
@邮箱: xingshunkai@qq.com
@网址: deepsenserobot.com
@B站: "阿凯爱玩机器人"
'''
# - 矩阵运算
import numpy as np
# - 图像处理
import cv2 
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335

# 创建相机对象
camera = Gemini335()
# 移除畸变
rm_distortion = False
# 创建窗口
win_flag = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
cv2.namedWindow("color", flags=win_flag)

# 指定视频编解码方式为MJPG
codec = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20.0 # 指定写入帧率为20
frameSize = (camera.img_width, camera.img_height) # 指定窗口大小
# 创建 VideoWriter对象
out = cv2.VideoWriter('video_record.avi', codec, fps, frameSize)
print("按键Q-结束视频录制")

while True:
	# 采集彩图, 色彩空间BGR
	img_bgr = camera.read_color_img() 
	# 去除彩图畸变
	if rm_distortion:
		img_bgr = camera.remove_distortion(img_bgr)
	# 不断的向视频输出流写入帧图像
	out.write(img_bgr)
	# 显示图像
	cv2.imshow('color', img_bgr)
	key = cv2.waitKey(1)
	if key == ord('q'):
		# 如果按键为q 代表quit 退出程序
		break
# 关闭视频流文件
out.release()
# 关闭摄像头
camera.release()
# 销毁所有的窗口
cv2.destroyAllWindows()
