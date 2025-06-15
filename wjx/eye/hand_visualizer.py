'''
ArucoTag可视化窗口(Astra相机版)
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import copy
import numpy as np
import open3d as o3d
# 阿凯机器人工具箱
from kyle_robot_toolbox.open3d import *
# 手掌3D可视化
from hand_o3d_gui import get_hand_geomery

class HandVisualizer():
	'''手掌关键点可视化窗口'''
	def __init__(self,  window_name="Hand"):
		# 窗口名字
		self.window_name = window_name
		# 可视化窗口
		self.visualizer = o3d.visualization.Visualizer()
		# 几何体列表
		self.geometry_list = []

	def create_window(self):
		'''创建窗口'''
		# 创建窗口
		self.visualizer.create_window(self.window_name, width=1280, height=720)
	
	def destroy_window(self):
		'''销毁窗口'''
		self.visualizer.destroy_window()

	def reset_scene_pcd(self):
		'''重置点云数据'''
		self.scene_pcd.clear()
	
	def reset_hand(self):
		'''清除已有的Hand相关的Geometry'''
		# 移除原有的Geometry
		for geometry in self.geometry_list:
			self.visualizer.remove_geometry(geometry)
		# 创建一个新的列表
		self.geometry_list = []

	def update_hand(self, point3d_list, is_reset=True):
		'''更新Hand相关的Geometry'''
		if is_reset:
			self.reset_hand()
		self.geometry_list = get_hand_geomery(point3d_list)
		for geometry in self.geometry_list:
			self.visualizer.add_geometry(geometry, reset_bounding_box=True)
	
	def step(self):
		'''更新一步'''
		# 接收事件
		self.visualizer.poll_events()
		# 渲染器需要更新
		self.visualizer.update_renderer()