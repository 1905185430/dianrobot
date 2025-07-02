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

camera = Gemini335()
processor = SingleImageProcessor()

img_bgr, depth_img = camera.read()
print("img_bgr.shape:", img_bgr.shape)
print("depth_img.shape:", depth_img.shape)
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test2.png", img_bgr)
# 转为np.uint16
depth_img = depth_img.astype(np.uint16)
depth_new = (depth_img.astype(np.uint16) * 255).astype(np.uint16)
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test2_depth.png", depth_new)

depth_canvas_tmp = camera.depth_img2canvas(depth_img)
dp_h, dp_w, dp_ch = depth_canvas_tmp.shape
depth_canvas = np.zeros_like(img_bgr)
depth_canvas[:dp_h, :dp_w] = depth_canvas_tmp
depth_canvas = (depth_canvas.astype(np.uint16) * 255).astype(np.uint16)
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test2_depth_new.png", depth_canvas)
cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/test2_depth_new2.png", depth_canvas_tmp)