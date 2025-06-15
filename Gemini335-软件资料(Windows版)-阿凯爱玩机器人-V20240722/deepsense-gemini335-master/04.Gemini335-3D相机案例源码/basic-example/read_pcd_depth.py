'''
Gemini335 3D相机 - 同时显示PCD点云与深度图
----------------------------------------------
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
# - 点云处理库
import open3d as o3d
# - Gemini335 3D相机
from kyle_robot_toolbox.camera import Gemini335
# - 点云动态刷新
from kyle_robot_toolbox.open3d import PCDVisualizer

# 创建相机对象
camera = Gemini335()

# 图像窗口
win_flag = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
cv2.namedWindow("depth", flags=win_flag)

# 点云动态可视化窗口
pcd_visual = PCDVisualizer()
pcd_visual.create_window()

while True:
	# 采集深度图
	depth_img = camera.read_depth_img()
	# 将深度图转换为画布
	canvas = camera.depth_img2canvas(depth_img, \
		min_distance=200, max_distance=1000)
	
	# 图像对齐，使用彩色相机的内参生成点云
	# 根据自己实际情况调整点云的最近距离(min_distance) 跟最远拍摄距离(max_distance)
	pcd = camera.get_pcd2(depth_img, \
		min_distance=200, max_distance=1000, \
		rgb_color=canvas)

	# 点云可视化窗口动态展示
	pcd_visual.update_pcd(pcd)
	# 可视化器迭代
	pcd_visual.step()

	# 深度图转换为画布
	depth_canvas = camera.depth_img2canvas(depth_img)
	# 显示图像
	cv2.imshow('depth', depth_canvas)
	key = cv2.waitKey(1)
	if key == ord('q'):
		# 如果按键为q 代表quit 退出程序
		break
# 关闭摄像头
camera.release()
# 销毁所有的窗口
cv2.destroyAllWindows()