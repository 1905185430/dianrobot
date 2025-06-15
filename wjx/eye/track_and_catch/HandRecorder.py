import numpy as np
import cv2
import os
from datetime import datetime
from HandDetector import HandDetector

class Hand:
    def __init__(self):
        # 初始化手部检测器
        self.detector = HandDetector()
        # 创建输出文件夹
        self.output_folder = "/home/xuan/dianrobot/wjx/hand_tracking_data"
        os.makedirs(self.output_folder, exist_ok=True)

        # 总体循环轨迹记录
        self.trajectory_2d = []  # 存储二维像素坐标轨迹
        # 轨迹记录变量
        self.recording = False
        self.trajectory_3d_record = []  # 存储三维世界坐标轨迹
        self.trajectory_obs_3d = []  # 存储三维世界坐标加姿态

        # 048点轨迹记录
        self.trajectory_dot0 = []
        self.trajectory_dot4 = []
        self.trajectory_dot8 = []

        self.max_trajectory_len = 1000  # 轨迹最大长度
        self.trajectory_file = None
        self.frame_count = 0
        self.stable = False
        self.stable_num = 0

        # 机械臂点到点夹取关键点检测值
        self.start_pos = None  # 起始位置
        self.end_pos = None  # 结束位置

        # demo检测手是否开始夹住物体
        self.hand_start = False
        self.hand_end = False

    def stable_or_not(self, memory=10, threshold=8):
        """
        检测手部位置是否稳定
        memory: 用于判断稳定性的历史帧数
        threshold: 方差阈值
        返回 True 表示稳定，False 表示不稳定
        """
        if len(self.trajectory_dot0) < memory:
            return False
        #trajectory_0 = np.array(self.trajectory_dot0[-memory:])
        trajectory_4 = np.array(self.trajectory_dot4[-memory:])
        #trajectory_8 = np.array(self.trajectory_dot8[-memory:])
        #var_0 = np.var(trajectory_0, axis=0)
        var_4 = np.var(trajectory_4, axis=0)
        #var_8 = np.var(trajectory_8, axis=0)
        if  np.all(var_4 < threshold) :
            print("手部位置稳定")
            #print(f"方差: dot4={var_4}")
            return True
        #print("手部位置不稳定")
        #print(f"方差: dot4={var_4}")
        return False

    def start_recording(self):
        self.recording = True
        self.frame_count = 0
        self.trajectory_3d_record = []
        print("开始记录手部轨迹")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.trajectory_file:
                self.trajectory_file.close()
                print(f"轨迹记录结束，共记录 {self.frame_count} 帧")
            

    def save_positions(self):
        if len(self.trajectory_3d_record) > 0:
            csv_filename = "/home/xuan/dianrobot/wjx/eye/hand_positions.csv"
            with open(csv_filename, "w") as f:
                f.write("world_x,world_y,world_z\n")
                for world_x, world_y, world_z in self.trajectory_3d_record:
                    f.write(f"{world_x:.2f},{world_y:.2f},{world_z:.2f}\n")
            print(f"手部位置数据已保存至 {csv_filename}")
            return csv_filename
        else:
            print("未记录到任何位置点")
            return None
        
    def save_positions_obs(self):
        if len(self.trajectory_obs_3d) > 0:
            csv_filename = "/home/xuan/dianrobot/wjx/eye/hand_obs.csv"
            with open(csv_filename, "w") as f:
                f.write("world_x,world_y,world_z,rx,ry,rz\n")
                for obs in self.trajectory_obs_3d:
                    if len(obs) == 6:
                        x, y, z, rx, ry, rz = obs
                    elif len(obs) == 3:
                        x, y, z = obs
                        rx, ry, rz = 0.0, 0.0, 0.0  # 或你需要的默认值
                    else:
                        continue  # 跳过异常数据
                    f.write(f"{x:.2f},{y:.2f},{z:.2f},{rx:.2f},{ry:.2f},{rz:.2f}\n")
            print(f"观察点数据已保存至 {csv_filename}")
            return csv_filename
        else:
            print("未记录到任何观察点")
            return None

    def update_trajectory(self):
        """
        获取手部关键点的世界坐标，并更新轨迹
        获取0、4、8号点的世界坐标,并写入轨迹

        """

        # 获取手部关键点的世界坐标
        if self.detector.dot_0_pos is not None:
            if self.detector.dot_8_pos is not None:
                if self.detector.dot_4_pos is not None:
                    self.trajectory_dot8.append(self.detector.dot_8_pos)
                    self.trajectory_dot0.append(self.detector.dot_0_pos)
                    self.trajectory_dot4.append(self.detector.dot_4_pos)
        if len(self.trajectory_dot0) > self.max_trajectory_len:
            self.trajectory_dot0.pop(0)
            self.trajectory_dot4.pop(0)
            self.trajectory_dot8.pop(0)
        # 记录世界坐标
        if self.recording:
            world_obj = self.detector.obj_pos_3d
            obj_2d = self.detector.obj_pos_2d
            if world_obj is not None:
                self.trajectory_3d_record.append(world_obj)
                if self.trajectory_file:
                    self.trajectory_file.write(f"{world_obj[0]:.2f},{world_obj[1]:.2f},{world_obj[2]:.2f}\n")
                self.frame_count += 1
                # 更新二维轨迹
                self.trajectory_2d.append(obj_2d)
                if len(self.trajectory_2d) > self.max_trajectory_len:
                    self.trajectory_2d.pop(0)
                self.trajectory_obs_3d.append(self.detector.obs_pos)
        

    def update_stable(self):
        """
        更新手部位置稳定性状态
        """
        if self.stable_or_not(memory=5, threshold=8):
            self.stable_num += 1
        else:
            self.stable_num = 0

        
    def update_canvas(self):
        """
        显示处理后的画布，包括手部关键点、世界坐标和2D轨迹
        """
        # 获取当前彩色图像
        color_img = self.detector.color_img
        results = self.detector.results
        px, py = self.trajectory_2d[-1] if self.trajectory_2d else (None, None)
        canvas = color_img.copy()
        # 绘制手部关键点和序号
        if results is not None and results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                self.detector.mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    self.detector.mp_hands.HAND_CONNECTIONS,
                    self.detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.detector.mp_drawing_styles.get_default_hand_connections_style())
            # 只显示第一个手的关键点序号
            hand_landmarks = results.multi_hand_landmarks[0]
            for i, mark in enumerate(hand_landmarks.landmark):
                px_i = int(mark.x * self.detector.camera.img_width)
                py_i = int(mark.y * self.detector.camera.img_height)
                cv2.putText(canvas, str(i), (px_i-25, py_i+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    

        # 绘制2D轨迹
        if len(self.trajectory_2d) > 1:
            for i in range(len(self.trajectory_2d) - 1):
                pt1 = self.trajectory_2d[i]
                pt2 = self.trajectory_2d[i + 1]
                color = (0, 255, 255)
                cv2.line(canvas, pt1, pt2, color, 2)

        # 绘制当前点
        if px is not None and py is not None:
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)
            cv2.circle(canvas, (px, py), 8, (0, 255, 255), 2)

        # 显示世界坐标
        pos_3d = self.detector.obj_pos_3d
       
        if pos_3d is not None:
            world_x, world_y, world_z = pos_3d
            world_text = f"World: X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}"

        # 显示self.detctor.obs_pos
        obs_pos = self.detector.obs_pos
        if obs_pos is not None:
            obs_x, obs_y, obs_z = obs_pos[:3]
            obs_text = f"Obs: X={obs_x:.2f}, Y={obs_y:.2f}, Z={obs_z:.2f}"
            if len(obs_pos) == 6:
                rx, ry, rz = obs_pos[3:6]
                obs_text += f", RX={rx:.2f}, RY={ry:.2f}, RZ={rz:.2f}"
            else:
                obs_text += ", RX=0.00, RY=0.00, RZ=0.00"
        else:
            obs_text = "Obs: None"
        if pos_3d is not None:
            cv2.putText(canvas, world_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(canvas, obs_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # 绘制0、4、8号点的轨迹


        cv2.imshow("Hand Detection", canvas)

    def updater(self):
        """
        更新手部检测器状态
        """
        self.detector.update()
        self.update_stable()
        self.update_trajectory()
        self.update_canvas()

    def run(self):
        """
        主循环：检测、判断稳定、记录、可视化
        """
        try:
            while True:
                # 更新手部检测器状态
                self.updater()
                if self.stable_num > 4 and not self.recording:
                    self.stable_num = 0  # 重置稳定计数
                    print("手部位置稳定，开始记录轨迹")
                    self.start_recording()
                if self.stable_num > 4 and self.recording and self.frame_count > 30:
                    self.stable_num = 0  # 重置稳定计数
                    print("手部位置持续稳定，停止记录轨迹")
                    self.stop_recording()
                    self.save_positions()
                    self.save_positions_obs()
                    break
                # 按键处理
                    
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            self.detector.camera_exit()
            cv2.destroyAllWindows()

    def run2(self):
        """
        主循环：检测、判断稳定、记录、可视化
        按R键开始/停止记录，ESC退出
        """
        print("按 R 开始/停止记录，ESC 退出")
        try:
            while True:
                # 更新手部检测器状态
                self.updater()

                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):
                    if not self.recording:
                        print("手动开始记录轨迹")
                        self.start_recording()
                    else:
                        print("手动停止记录轨迹")
                        self.stop_recording()
        finally:
            self.detector.camera_exit()
            cv2.destroyAllWindows()

    

if __name__ == "__main__":
    hand = Hand()
    hand.run2()