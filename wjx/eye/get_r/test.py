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

rgb_img = camera.read_color_img()
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test.png", rgb_img)
depth_img = camera.read_depth_img()
print(type(depth_img))
print(depth_img.shape)
print(depth_img.dtype)
depth_img = depth_img.astype(np.uint16)
print(depth_img.shape)
print(depth_img.dtype)
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test_depth.png", depth_img)