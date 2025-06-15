'''
Gemini335 3D相机 - 同时显示PCD点云与彩图+深度图
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

# 创建相机 指定SN码(串口设备号)
# 注: SN码可以通过上位机查询，每一个设备都有独立的SN码
# camera = Gemini2(serial_num="AY3A131006E")

# 图像窗口
win_flag = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
cv2.namedWindow("color", flags=win_flag)
cv2.namedWindow("depth", flags=win_flag)

# 点云动态可视化窗口
pcd_visual = PCDVisualizer()
pcd_visual.create_window()

img_cnt = 1
while True:
	# 采集彩图, 色彩空间BGR
	color_img, depth_img = camera.read()
	if color_img is None or depth_img is None:
		print("图像获取失败")
		continue
 	# 去除彩图畸变
	color_img = camera.remove_distortion(color_img)
	# 获取彩色点云
	# 根据自己实际情况调整点云的最近距离(min_distance) 跟最远拍摄距离(max_distance)
	pcd = camera.get_pcd(color_img, depth_img,\
		min_distance=200, max_distance=1000, \
		camera="rgb_camera")
	
	# 点云可视化窗口动态展示
	pcd_visual.update_pcd(pcd)
	# 可视化器迭代
	pcd_visual.step()

	# 深度图转换为画布
	depth_canvas = camera.depth_img2canvas(depth_img, \
		min_distance=200, max_distance=1000)
	# 显示图像
	cv2.imshow('color', color_img)
	cv2.imshow('depth', depth_canvas)
	key = cv2.waitKey(1)
	if key == ord('q'):
		# 如果按键为q 代表quit 退出程序
		break
	elif key == ord('s'):
		print(f"保存图像跟点云，ID={img_cnt}")
		print("存储路径: data/capture")
		# 保存彩图+深度图+点云
		# - 保存彩图
		cv2.imwrite(f"data/capture/{img_cnt}.png", color_img)
		# - 保存深度图
		np.save(f"data/capture/{img_cnt}.npy", depth_img)
		# - 保存点云
		o3d.io.write_point_cloud(f"./data/capture/{img_cnt}.pcd", pcd)
		img_cnt += 1
# 关闭摄像头
camera.release()
# 销毁所有的窗口
cv2.destroyAllWindows()