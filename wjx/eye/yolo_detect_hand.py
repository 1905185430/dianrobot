'''
手掌关键点检测与定位
- 获取手掌0号点在相机坐标系下的三维坐标
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import json
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335
from kyle_robot_toolbox.open3d import *
# 自定义库
# - 手掌可视化库

def depth_to_pointcloud(color_img, depth_img, intrinsic):
    """
    将深度图和彩色图转换为点云
    :param color_img: 彩色图 (H, W, 3), uint8
    :param depth_img: 深度图 (H, W), uint16 或 float32，单位mm或m
    :param intrinsic: 相机内参，open3d.camera.PinholeCameraIntrinsic 或 numpy数组
    :return: open3d.geometry.PointCloud
    """
    # 如果 intrinsic 是 numpy 数组，转为 open3d 格式
    if isinstance(intrinsic, np.ndarray):
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        width, height = color_img.shape[1], color_img.shape[0]
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    else:
        o3d_intrinsic = intrinsic

    # 转为 open3d 图像格式
    color_o3d = o3d.geometry.Image(color_img)
    # 若深度单位为mm，需转为m
    if depth_img.dtype != np.float32:
        depth_o3d = o3d.geometry.Image((depth_img / 1000.0).astype(np.float32))
    else:
        depth_o3d = o3d.geometry.Image(depth_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d_intrinsic)
    return pcd

# 创建相机对象
camera = Gemini335()


intrinsic = camera.get_intrinsic()
camera_to_world = np.array([
        [0.9995235405012892, 0.0026994478399852235, -0.03074743835064323, -28.244393511054284],
        [0.003988141025453063, -0.999112661826739, 0.04192831631069003, -701.4038878017706],
        [-0.030606971671965612, -0.04203096428643264, -0.9986473908874063, 845.7511974648796],
        [0.0, 0.0, 0.0, 1.0]
    ])
# min_detection_confidence: 置信度阈值 
# min_tracking_confidence: 跟踪阈值
# 获取彩图
color_img = camera.read_color_img()
# 获取深度图
depth_img = camera.read_depth_img()
# 生成画布
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3)

# 绘图
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
while True:
    
    # 采集图像
    # 注: 前几帧图像质量不好，可以多采集几次  
    # 另外确保画面中有手
    img_bgr = camera.read_color_img()
    depth_img = camera.read_depth_img()
    if depth_img.shape[:2] != img_bgr.shape[:2]:
        depth_img = cv2.resize(depth_img, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 为了提高性能， 将图像标记为只读模式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    # 手掌关键点检测
    results = hands.process(img_rgb)
    # 创建画布
    canvas = np.copy(img_bgr)
    if results.multi_hand_landmarks is not None:
        # 绘制关键点
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # 绘制关键点序号
        for hand_landmarks in results.multi_hand_landmarks:    
            for i, mark in enumerate(hand_landmarks.landmark):
                px = int(mark.x * camera.img_width)
                py = int(mark.y * camera.img_height)
                cv2.putText(canvas, str(i), (px-25, py+5), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    

    pcd = depth_to_pointcloud(color_img=img_bgr, 
                              depth_img=depth_img, 
                              intrinsic=intrinsic)
    



    if results.multi_hand_landmarks is not None:
        pixel_landmark = results.multi_hand_landmarks[0].landmark
        # 获取Landmark的像素坐标
        # 获取关键点4,8的像素坐标取平均
        px_4 = int(pixel_landmark[4].x * camera.img_width)
        py_4 = int(pixel_landmark[4].y * camera.img_height)
        px_8 = int(pixel_landmark[8].x * camera.img_width)
        py_8 = int(pixel_landmark[8].y * camera.img_height)
        px = int((px_4 + px_8) / 2)
        py = int((py_4 + py_8) / 2)
        # 在canvas上标注px,py点
        dp_h, dp_w = depth_img.shape
        #根据像素坐标获取深度值
      
        roi_size = 5
        y_min = max(0, py-roi_size)
        y_max = min(dp_h, py+roi_size)
        x_min = max(0, px-roi_size)
        x_max = min(dp_w, px+roi_size)
        
        # 获取ROI区域
        roi = depth_img[y_min:y_max, x_min:x_max]
        
        # 过滤无效深度值
        index = np.where(roi != 0)
        if len(index[0]) > 0:  # 确保有有效数据点
            depth_value = np.mean(roi[index])
            print(f"深度值: {depth_value:.1f}mm")
            # 将像素坐标转换为相机坐标系下的三维坐标
            cam_point3d = camera.depth_pixel2cam_point3d(
                px, py, depth_value=depth_value, intrinsic=intrinsic)
            cam_x, cam_y, cam_z = cam_point3d
            
            # 显示坐标信息
            info_text = f"3D: [{cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}]mm"
            print(f"彩色相机坐标系下的坐标: [{cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}], 单位mm")
            
            # 绘制指示标记
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)  # 实心黄色圆点
            cv2.circle(canvas, (px, py), 8, (0, 255, 255), 2)   # 空心黄色圆环
            
            # 在画面上显示坐标信息
            cv2.putText(canvas, info_text, (px+10, py-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 在ROI区域周围绘制矩形框
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    cv2.imshow("Hand Detection", canvas)
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == ord('p'):  # 按p键可视化点云
        pcd = depth_to_pointcloud(color_img=img_bgr, 
                                  depth_img=depth_img, 
                                  intrinsic=intrinsic)
        o3d.visualization.draw_geometries([pcd])
        depth_vis = depth_img.copy()
        if depth_vis.dtype != np.uint8:
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
        cv2.imshow("Depth", depth_vis)

        cv2.imshow("Hand Detection", canvas)
        key = cv2.waitKey(5) & 0xFF
    if cv2.waitKey(5) & 0xFF == 27:
        break
camera.release()
