'''
Gemini335 3D相机 - 测试彩图读取与动态更新
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
# - Open3D点云处理
import open3d as o3d
# 自定义库
# 从阿凯机器人工具箱导入Gemini335类
from kyle_robot_toolbox.camera import Gemini335

# 创建相机对象
camera = Gemini335()

# 创建窗口
win_flag = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
cv2.namedWindow("color", flags=win_flag)

img_cnt = 0
while True:
	# 采集彩图, 色彩空间BGR
	img_bgr = camera.read_color_img() 
	if img_bgr is None:
		print("彩图获取失败")
		continue
	# 显示图像
	cv2.imshow('color', img_bgr)
	key = cv2.waitKey(1)
	if key == ord('q'):
		# 如果按键为q 代表quit 退出程序
		break
	elif key == ord('s'):
		img_path = f"data/capture/{img_cnt}.png"
		cv2.imwrite(img_path, img_bgr)
		img_cnt += 1
		print(f"保存图像: {img_path}")

# 关闭摄像头
camera.release()
# 销毁所有的窗口
cv2.destroyAllWindows()