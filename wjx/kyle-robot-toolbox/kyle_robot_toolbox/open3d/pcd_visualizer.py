'''
点云动态刷新窗口
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import copy
import numpy as np
import open3d as o3d

class PCDVisualizer():
	'''点云可视化窗口'''
	def __init__(self, window_name="PCD") -> None:
		# 窗口名字
		self.window_name = window_name
		# 可视化窗口
		self.visualizer = o3d.visualization.Visualizer()
		# 点云数据
		self.pcd = o3d.geometry.PointCloud()
		# 是否添加了Geometry
		self.add_geomery = False
	
	def create_window(self):
		'''创建窗口'''
		# 创建窗口
		self.visualizer.create_window(self.window_name, width=1280, height=720)
	
	def destroy_window(self):
		'''销毁窗口'''
		self.visualizer.destroy_window()
	
	def reset_pcd(self):
		'''重置点云数据'''
		self.pcd.clear()
	
	def update_pcd(self, pcd, is_reset=True):
		'''更新点云数据'''
		if is_reset:
			self.reset_pcd()
		# 添加新的PCD
		self.pcd += copy.deepcopy(pcd)
		if not self.add_geomery:
			self.visualizer.add_geometry(self.pcd)
			self.add_geomery = True
		# 可视化窗口更新
		self.visualizer.update_geometry(self.pcd)
	
	def step(self):
		'''更新一步'''
		# 接收事件
		self.visualizer.poll_events()
		# 渲染器需要更新
		self.visualizer.update_renderer()

# def main(argv):
# 	import cv2
# 	import time
# 	# 自定义库
# 	from astra import Astra

# 	# 创建Realsense
# 	# 创建相机对象
# 	camera = Astra()
# 	# 初始相机
# 	camera.init_video_stream(\
# 	 	video_mode=FLAGS.video_mode, \
# 		image_registration=FLAGS.image_registration)
# 	print(f"视频流模式:  {FLAGS.video_mode}")
# 	print(f"是否开启深度图对齐: {FLAGS.image_registration}")
# 	print(f"是否移除畸变: {FLAGS.rm_distortion}")
# 	# 读取相机参数(相机内参)
# 	camera.load_cam_calib_data()
	
# 	# 创建可视化窗口
# 	visualizer = PCDVisualizer()
# 	visualizer.create_window()

# 	pcd = None
# 	print("开始拍摄")
# 	try:
# 		while True:
# 			if FLAGS.video_mode == "color_depth":
# 				# 获取彩图与深度图
# 				color_image = camera.read_color_img(adjust_color_image=False)
# 				# print("拍摄彩图")
# 				depth_image = camera.read_depth_img()
# 				# 移除图像畸变
# 				if FLAGS.rm_distortion:
# 					color_image = camera.rgb_camera.remove_distortion(color_image)
# 				# 转换为点云
# 				# pcd = camera.rgb_camera.get_pcd(color_image, depth_image,\
# 				# 	o3d_intrinsic=camera.rgb_o3d_intrinsic_shift)
# 				pcd = camera.rgb_camera.get_pcd(color_image, depth_image)
# 				# print("转换为PCD点云")
# 			elif FLAGS.video_mode == "depth":
# 				# 获取深度图
# 				depth_image = camera.read_depth_img()
# 				# 转换为点云
# 				# pcd = camera.get_pcd2(depth_image, rgb_color=[125, 125, 125])
# 				if FLAGS.image_registration:
# 					pcd = camera.rgb_camera.get_pcd2(depth_image)
# 				else:
# 					pcd = camera.ir_camera.get_pcd2(depth_image)
# 			elif FLAGS.video_mode == "ir_depth":
# 				# 获取IR图(平移后的IR图)
# 				ir_image = camera.read_ir_img(adjust_ir_image=True)
# 				# 获取深度图
# 				depth_image = camera.read_depth_img()
# 				# 根据IR图与深度图获取点云
# 				pcd = camera.ir_camera.get_pcd(ir_image, depth_image, \
# 					o3d_intrinsic=camera.ir_o3d_intrinsic_shift)

# 				# 获取点云
# 				# pcd = camera.ir_camera.get_pcd(ir_image, depth_image)
# 				# 修正深度图
# 				# depth_image2 = np.copy(depth_image)
# 				# depth_image2 = camera.adjust_depth_img_from_ir(depth_image2)
# 				# 注: 此时这里的深度图，也还是错误的深度
# 				# 重新映射? 做出来新的深度图?
# 				# depth_image2 = np.zeros_like(depth_image, dtype=np.float32)
# 				# v_offset = 10
# 				# depth_image2[:-v_offset, :] = depth_image[v_offset:, :] 
# 				# pcd = camera.ir_camera.get_pcd(ir_image, depth_image2)
			
# 			# 更新可PCD视化器里面的
# 			visualizer.update_pcd(pcd)
# 			# 可视化器迭代
# 			visualizer.step()
# 			time.sleep(0.01)
# 	except Exception as e:
# 		# 关闭窗口
# 		visualizer.destroy_window()
# 		# 释放相机
# 		camera.release()


# if __name__ == '__main__':
# 	import logging
# 	import sys
# 	from absl import app
# 	from absl import flags

# 	# 设置日志等级
# 	logging.basicConfig(level=logging.INFO)

# 	# 定义参数
# 	FLAGS = flags.FLAGS
# 	flags.DEFINE_string('video_mode', 'color_depth', '视频流类型')
# 	flags.DEFINE_boolean('rm_distortion', False, '载入相机标定数据, 去除图像畸变')
# 	flags.DEFINE_boolean('image_registration', True, '深度图与彩图对齐')
# 	try:
# 		# 运行主程序
# 		app.run(main)
# 	except Exception as e:
# 		print(e)
	