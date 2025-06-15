import numpy as np
import cv2
import os
from datetime import datetime
from HandDetector import HandDetector

# 创建输出文件夹
output_folder = "/home/xuan/dianrobot/wjx/hand_tracking_data"
os.makedirs(output_folder, exist_ok=True)

# 轨迹记录变量
recording = False
trajectory_2d = []  # 存储二维像素坐标轨迹
trajectory_3d = []  # 存储三维世界坐标轨迹
max_trajectory_len = 1000  # 轨迹最大长度
trajectory_file = None
frame_count = 0

# 初始化手部检测器
detector = HandDetector()
# Height_bias
height_bias = 90

def start_recording():
    global recording, trajectory_file, trajectory_2d, trajectory_3d, frame_count
    recording = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_folder}/hand_trajectory_{timestamp}.csv"
    trajectory_file = open(csv_filename, "w")
    trajectory_file.write("frame,px,py,world_x,world_y,world_z\n")
    frame_count = 0
    trajectory_2d = []
    trajectory_3d = []
    print(f"开始记录手部轨迹到 {csv_filename}")

def stop_recording():
    global recording, trajectory_file
    if recording:
        recording = False
        if trajectory_file:
            trajectory_file.close()
            print(f"轨迹记录结束，共记录 {frame_count} 帧")
        save_positions()

def save_positions():
    if len(trajectory_3d) > 0:
        txt_filename = "/home/xuan/dianrobot/wjx/eye/hand_positions.txt"
        with open(txt_filename, "w") as f:
            for i, (world_x, world_y, world_z) in enumerate(trajectory_3d):
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
    # 获取检测结果和世界坐标
    detector.color_img = detector.camera.read_color_img()
    detector.results = detector.hands.process(cv2.cvtColor(detector.color_img, cv2.COLOR_BGR2RGB))
    world_point = detector.calculate_obj_position()
    canvas = np.copy(detector.color_img)

    px, py = None, None
    if detector.results.multi_hand_landmarks is not None:
        # 绘制关键点和序号
        for hand_landmarks in detector.results.multi_hand_landmarks:
            detector.mp_drawing.draw_landmarks(
                canvas,
                hand_landmarks,
                detector.mp_hands.HAND_CONNECTIONS,
                detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                detector.mp_drawing_styles.get_default_hand_connections_style())
            for i, mark in enumerate(hand_landmarks.landmark):
                px = int(mark.x * detector.camera.img_width)
                py = int(mark.y * detector.camera.img_height)
                cv2.putText(canvas, str(i), (px-25, py+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 只记录第一个手的4、8号点中点
        pixel_landmark = detector.results.multi_hand_landmarks[0].landmark
        px_4 = int(pixel_landmark[4].x * detector.camera.img_width)
        py_4 = int(pixel_landmark[4].y * detector.camera.img_height)
        px_8 = int(pixel_landmark[8].x * detector.camera.img_width)
        py_8 = int(pixel_landmark[8].y * detector.camera.img_height)
        px = int((px_4 + px_8) / 2)
        py = int((py_4 + py_8) / 2)

        # 绘制指示标记
        cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)
        cv2.circle(canvas, (px, py), 8, (0, 255, 255), 2)

        # 显示世界坐标
        if world_point is not None:
            world_x, world_y, world_z = world_point
            world_text = f"世界: [{world_x:.1f}, {world_y:.1f}, {world_z:.1f}]mm"
            cv2.putText(canvas, world_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 记录轨迹
            if recording and (world_z > height_bias):
                frame_count += 1
                trajectory_file.write(f"{frame_count},{px},{py},{world_x:.1f},{world_y:.1f},{world_z:.1f}\n")
                trajectory_file.flush()
                trajectory_2d.append((px, py))
                trajectory_3d.append((world_x, world_y, world_z))
                if len(trajectory_2d) > max_trajectory_len:
                    trajectory_2d.pop(0)
                    trajectory_3d.pop(0)

    # 绘制历史轨迹
    if len(trajectory_2d) > 1:
        for i in range(len(trajectory_2d)-1):
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
        if not recording:
            start_recording()
        else:
            stop_recording()
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{output_folder}/frame_{timestamp}.png"
        cv2.imwrite(save_path, canvas)
        print(f"截图已保存至 {save_path}")
        save_positions()

# 结束录制
stop_recording()
detector.camera_exit()
cv2.destroyAllWindows()