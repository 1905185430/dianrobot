'''
Gemini335 3D相机 - 测试深度图读取与动态更新
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
cv2.namedWindow("depth", flags=win_flag)

while True:
	# 采集深度图
	depth_img = camera.read_depth_img() 
	
	# 将深度图转换为画布
	canvas = camera.depth_img2canvas(depth_img)
	# 显示图像
	cv2.imshow('depth', canvas)
	key = cv2.waitKey(1)
	if key == ord('q'):
		# 如果按键为q 代表quit 退出程序
		break

# 关闭摄像头
camera.release()
# 销毁所有的窗口
cv2.destroyAllWindows()