'''
手掌检测与定位
'''
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
import os
import time
from datetime import datetime
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335
from SingleImageProsessor import SingleImageProcessor
from fairino import Robot

camera = Gemini335()
processor = SingleImageProcessor()
robot = Robot.RPC('192.168.58.6')
input("拍照")

rgb_img = camera.read_color_img()
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test.png", rgb_img)
depth_img = camera.read_depth_img()
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test_depth.png", depth_img)
# while pos is None:
#     img = camera.read_color_img()
#     cv2.imwrite("test.jpg", img)
#     processor.rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     depth_img = camera.read_depth_img()
#     processor.depth_image = depth_img
#     processor.images_to_results()
#     processor.update_all_axes()
#     processor.get_tool2world_transformation()
#     processor.draw_workpiece_axes_on_image(axis_length=50, show=True, window_name="Workpiece Axes")
#     processor.get_rxryrz_from_rotation_matrix()
#     processor.update_gripper_position()
#     pos = processor.gripper_pos
# print("抓取位姿:", pos)
# 复位
pos = processor.run()
pos = np.array(pos) + np.array([0, 100, 0, 0, 0, 0])
input("复位")
robot.MoveCart([0, -380, 200, 90, 0, 0], 0, 0, vel=20, acc=20)
input("复现位姿")
robot.MoveCart(pos, 0, 0, vel=20, acc=20)
img = camera.read_color_img()
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/机械臂复现.png", img)
input("复位")
robot.MoveCart([0, -380, 200, 90, 0, 0], 0, 0, vel=20, acc=20)








