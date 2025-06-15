'''
手掌关键点检测与定位
- 获取手掌0号点在相机坐标系下的三维坐标
- 支持记录手部运动轨迹
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

# 创建输出文件夹
output_folder = "/home/xuan/dianrobot/wjx/hand_tracking_data"
os.makedirs(output_folder, exist_ok=True)

# 创建相机对象
camera = Gemini335()
intrinsic = camera.get_intrinsic()
camera_to_world = np.array([
        [0.9995235405012892, 0.0026994478399852235, -0.03074743835064323, -28.244393511054284],
        [0.003988141025453063, -0.999112661826739, 0.04192831631069003, -701.4038878017706],
        [-0.030606971671965612, -0.04203096428643264, -0.9986473908874063, 845.7511974648796],
        [0.0, 0.0, 0.0, 1.0]
    ])

# 轨迹记录变量
recording = False
trajectory_2d = []  # 存储二维像素坐标轨迹
trajectory_3d = []  # 存储三维世界坐标轨迹
max_trajectory_len = 100  # 轨迹最大长度，防止内存占用过大
trajectory_file = None
frame_count = 0

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

# 转换函数：相机坐标到世界坐标
def transform_to_world_coordinates(point3d):
    """
    将相机坐标系中的点转换到世界坐标系
    
    参数:
        point3d: 相机坐标系下的点 [x, y, z]
        
    返回:
        world_point: 世界坐标系下的点 [x, y, z]
    """
    # 创建齐次坐标
    cam_point = np.append(point3d, 1)
    # 转换到世界坐标系
    world_point = np.dot(camera_to_world, cam_point)
    return world_point[:3]

# 开始记录
def start_recording():
    global recording, trajectory_file, trajectory_2d, trajectory_3d, frame_count
    recording = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建CSV文件记录详细数据
    csv_filename = f"{output_folder}/hand_trajectory_{timestamp}.csv"
    trajectory_file = open(csv_filename, "w")
    trajectory_file.write("frame,px,py,cam_x,cam_y,cam_z,world_x,world_y,world_z\n")
    
    frame_count = 0
    trajectory_2d = []
    trajectory_3d = []
    print(f"开始记录手部轨迹到 {csv_filename}")

# 停止记录并保存
def stop_recording():
    global recording, trajectory_file
    if recording:
        recording = False
        if trajectory_file:
            trajectory_file.close()
            print(f"轨迹记录结束，共记录 {frame_count} 帧")
            
        # 保存位置数据为TXT格式
        save_positions()

# 保存为txt格式
def save_positions():
    if len(trajectory_3d) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = "/home/xuan/dianrobot/wjx/eye/hand_positions.txt"
        
        with open(txt_filename, "w") as f:
            for i, (world_x, world_y, world_z) in enumerate(trajectory_3d):
                # 使用与aruco_positions.txt相同的格式
                f.write(f"检测 {i+1}: X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}\n")
        
        print(f"手部位置数据已保存至 {txt_filename}")
        return txt_filename
    else:
        print("未记录到任何位置点")
        return None

print("按 'r' 开始/停止记录轨迹")
print("按 's' 保存当前轨迹为TXT")
print("按 'ESC' 退出程序")

while True:
    # 采集图像
    # 注: 前几帧图像质量不好，可以多采集几次  
    # 另外确保画面中有手
    img_bgr = camera.read_color_img()
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
        depth_img = camera.read_depth_img()
        if depth_img is None:
            print("[WARN] 深度数据帧获取失败，跳过当前帧处理")
            # 显示错误信息在画面上
            cv2.putText(canvas, "深度数据获取失败!", (10, canvas.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Hand Detection", canvas)
            continue
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
            depth_value = np.mean(roi[index])  # 计算有效深度值的平均值
            
            # 将像素坐标转换为相机坐标系下的三维坐标
            cam_point3d = camera.depth_pixel2cam_point3d(
                px, py, depth_value=depth_value, intrinsic=intrinsic)
            cam_x, cam_y, cam_z = cam_point3d
            
            # 转换到世界坐标系
            world_point = transform_to_world_coordinates([cam_x, cam_y, cam_z])
            world_x, world_y, world_z = world_point
            
            # 显示坐标信息
            info_text = f"3D: [{cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}]mm"
            world_text = f"世界: [{world_x:.1f}, {world_y:.1f}, {world_z:.1f}]mm"
            
            # 绘制指示标记
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)  # 实心黄色圆点
            cv2.circle(canvas, (px, py), 8, (0, 255, 255), 2)   # 空心黄色圆环
            
            # 在画面上显示坐标信息
            cv2.putText(canvas, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(canvas, world_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 在ROI区域周围绘制矩形框
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            
            # 记录轨迹数据
            if recording:
                frame_count += 1
                # 记录到文件
                trajectory_file.write(f"{frame_count},{px},{py},{cam_x:.1f},{cam_y:.1f},{cam_z:.1f},{world_x:.1f},{world_y:.1f},{world_z:.1f}\n")
                trajectory_file.flush()  # 确保立即写入磁盘
                
                # 添加到轨迹列表
                trajectory_2d.append((px, py))
                trajectory_3d.append((world_x, world_y, world_z))
                
                # 限制轨迹长度
                if len(trajectory_2d) > max_trajectory_len:
                    trajectory_2d.pop(0)
                    trajectory_3d.pop(0)
            
            # 绘制历史轨迹
            if len(trajectory_2d) > 1:
                for i in range(len(trajectory_2d)-1):
                    # 根据点的时间先后，颜色从浅绿到深绿逐渐变化
                    color_factor = i / len(trajectory_2d)
                    color = (0, 255, int(255 * (1-color_factor)))
                    cv2.line(canvas, trajectory_2d[i], trajectory_2d[i+1], color, 2)
    
    # 显示记录状态
    if recording:
        cv2.putText(canvas, f"记录中: 已记录 {frame_count} 帧", (10, canvas.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Hand Detection", canvas)

    # 按键处理
    key = cv2.waitKey(5)
    if key & 0xFF == 27:  # ESC键
        break
    elif key == ord('r'):
        # 开始/停止记录
        if not recording:
            start_recording()
        else:
            stop_recording()
    elif key == ord('s'):
        # 保存当前帧和轨迹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{output_folder}/frame_{timestamp}.png"
        cv2.imwrite(save_path, canvas)
        print(f"截图已保存至 {save_path}")
        save_positions()

# 结束录制
stop_recording()
# 释放资源
camera.release()
cv2.destroyAllWindows()