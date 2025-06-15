'''
ArucoTag检测+位姿估计(3D相机版) - 无ROS版本
----------------------------
修改：移除ROS依赖，使用相机到世界坐标系转换矩阵计算物体位置
'''
import time
import json
import numpy as np
import cv2
import os
# 阿凯机器人工具箱
# - Gemini335类
from kyle_robot_toolbox.camera import Gemini335
from kyle_robot_toolbox.opencv import ArucoTag
from kyle_robot_toolbox.open3d import *
# ArucoTag可视化窗口
from arucotag_visualizer import ArucoTagVisualizer
# ArucoTag姿态矫正器
from arucotag_pose_adjust import *

class ObjectTracker:
    def __init__(self, camera_to_world_matrix=None, output_file="object_positions.txt"):
        """
        初始化物体跟踪器
        
        参数:
            camera_to_world_matrix: 相机到世界坐标系的4x4变换矩阵
            output_file: 输出文件路径
        """
        # 相机到世界坐标系的变换矩阵
        self.camera_to_world = camera_to_world_matrix if camera_to_world_matrix is not None else np.eye(4)
        
        # 输出文件
        self.output_file = output_file
        self.recording = False
        self.recorded_positions = []
        self.detection_count = 0
        
        # 创建相机对象
        self.camera = Gemini335()
        
        # 创建ArucoTag检测器
        self.arucotag = ArucoTag(self.camera, 
            config_path="/home/xuan/dianrobot/wjx/eye/arucotag/arucotag.yaml")
        
        # 创建ArucoTag可视化窗口
        aruco_size = self.arucotag.config["aruco_size"]/1000.0
        box_depth = 0.01
        self.visualizer = ArucoTagVisualizer(self.camera, 
            aruco_size=aruco_size, 
            box_depth=box_depth)
        self.visualizer.create_window()
        
        # 配置视角
        json_path = "/home/xuan/dianrobot/wjx/eye/arucotag/render_option.json"
        trajectory = json.load(open(json_path, "r", encoding="utf-8"))
        self.view_point = trajectory["trajectory"][0]
        self.is_draw_camera = False
    
    def set_view_control(self):
        '''控制视野'''
        ctr = self.visualizer.visualizer.get_view_control()
        ctr.set_front(np.float64(self.view_point["front"]))
        ctr.set_lookat(np.float64(self.view_point["lookat"]))
        ctr.set_up(np.float64(self.view_point["up"]))
        ctr.set_zoom(np.float64(self.view_point["zoom"]))
    
    def transform_to_world_coordinates(self, cam_matrix):
        """
        将相机坐标系中的变换矩阵转换到世界坐标系
        
        参数:
            cam_matrix: 物体在相机坐标系的4x4变换矩阵
            
        返回:
            world_matrix: 物体在世界坐标系的4x4变换矩阵
        """
        # 计算物体在世界坐标系的变换矩阵
        world_matrix = np.dot(self.camera_to_world, cam_matrix)
        return world_matrix
    
    def start_recording(self):
        """开始记录物体位置"""
        self.recording = True
        self.recorded_positions = []
        self.detection_count = 0
        print("开始记录物体位置...")
    
    def stop_recording(self):
        """停止记录并保存结果"""
        self.recording = False
        if len(self.recorded_positions) > 0:
            self.save_positions()
            print(f"记录结束，共记录 {len(self.recorded_positions)} 个位置点")
        else:
            print("未记录到任何位置点")
    
    def save_positions(self):
        """保存记录的位置到文件"""
        try:
            # 确保路径存在
            dir_path = os.path.dirname(self.output_file)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            with open(self.output_file, 'w') as f:
                for i, pos in enumerate(self.recorded_positions):
                    x, y, z, rx, ry, rz = pos
                    f.write(f"检测 {i+1}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, RX={rx:.2f}, RY={ry:.2f}, RZ={rz:.2f}\n")
            print(f"位置数据已保存到 {self.output_file}")
            return True
        except Exception as e:
            print(f"保存位置数据失败: {e}")
            return False
    
    def extract_position_and_rotation(self, transform_matrix):
        """
        从变换矩阵中提取位置和旋转角度
        
        参数:
            transform_matrix: 4x4变换矩阵
            
        返回:
            (x, y, z, rx, ry, rz): 位置和欧拉角(度)
        """
        # 提取位置
        x = transform_matrix[0, 3]
        y = transform_matrix[1, 3]
        z = transform_matrix[2, 3]
        
        # 提取旋转矩阵
        R = transform_matrix[:3, :3]
        
        # 将旋转矩阵转换为欧拉角 (ZYX顺序)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:  # 非奇异情况
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:  # 奇异情况
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
        
        # 转换为角度
        rx_deg = np.degrees(rx)
        ry_deg = np.degrees(ry)
        rz_deg = np.degrees(rz)
        
        return (x, y, z, rx_deg, ry_deg, rz_deg)
    
    def run(self):
        """主循环"""
        print("按 'r' 开始/停止记录位置")
        print("按 's' 保存当前记录的位置")
        print("按 'q' 退出程序")
        
        try:
            while True:
                # 采集图像
                img_bgr, depth_img = self.camera.read()
                # 图像移除畸变
                img_bgr = self.camera.remove_distortion(img_bgr)
                # 图像预处理
                img_filter = image_preprocessor(img_bgr)
                # 根据深度图生成画布
                depth_canvas = self.camera.depth_img2canvas(depth_img, 
                    min_distance=150, max_distance=300)
                # 彩图+深度图生成点云
                scene_pcd = self.camera.get_pcd(img_bgr, depth_img)
                
                # 更新可视化窗口：场景点云
                self.visualizer.update_scene_pcd(scene_pcd)
                
                # ArucoTag检测
                has_aruco, canvas, aruco_ids, aruco_centers, \
                aruco_corners, T_cam2aruco_by2d = \
                    self.arucotag.aruco_pose_estimate(img_filter)
                
                cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
                T_cam2aruco_by3d_filter = []
                
                if has_aruco:
                    # 矫正ArucoTag坐标
                    valid_aruco_mask, t_cam2aruco_by3d_filter = get_t_cam2aruco_by3d( 
                        self.camera, depth_img, aruco_ids, aruco_centers, 
                        canvas=canvas, depth_canvas=depth_canvas)
                    
                    # 过滤有效的数据
                    aruco_ids_filter = aruco_ids[valid_aruco_mask]
                    aruco_centers_filter = aruco_centers[valid_aruco_mask]
                    aruco_corners_filter = aruco_corners[valid_aruco_mask]
                    T_cam2aruco_by2d_filter = T_cam2aruco_by2d[valid_aruco_mask]
                    
                    # 姿态矫正
                    T_cam2aruco_by3d_filter = adjust_T_cam2aruco(
                        self.camera, img_filter, depth_img,
                        aruco_ids_filter, aruco_corners_filter,
                        T_cam2aruco_by2d_filter, t_cam2aruco_by3d_filter)
                    
                    # 从T_cam2aruco_by3d_filter中提取位置信息
                    if len(T_cam2aruco_by3d_filter) > 0:
                        # 显示标签ID
                        for i, aruco_id in enumerate(aruco_ids_filter):
                            cv2.putText(canvas, f"ID: {aruco_id}", 
                                        tuple(aruco_centers_filter[i].astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 提取第一个标签的变换矩阵
                        cam_matrix = T_cam2aruco_by3d_filter[0]
                        
                        # 转换到世界坐标系
                        world_matrix = self.transform_to_world_coordinates(cam_matrix)
                        
                        # 提取位置和旋转信息
                        x, y, z, rx, ry, rz = self.extract_position_and_rotation(world_matrix)
                        
                        # 在画面上显示世界坐标
                        text_pos = (50, 50)
                        cv2.putText(canvas, f"World: X={x:.2f}, Y={y:.2f}, Z={z:.2f}", 
                                   text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(canvas, f"Rot: RX={rx:.2f}, RY={ry:.2f}, RZ={rz:.2f}", 
                                   (text_pos[0], text_pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 如果正在记录，添加到记录列表
                        if self.recording:
                            self.detection_count += 1
                            # 每5帧记录一次，避免记录太多相似的点
                            if self.detection_count % 5 == 0:
                                self.recorded_positions.append((x, y, z, rx, ry, rz))
                                print(f"记录位置点 {len(self.recorded_positions)}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
                    
                    # 更新ArucoTag的可视化模型
                    self.visualizer.update_aruco(T_cam2aruco_by3d_filter)
                else:
                    # 没检测到ArucoTag时，复位显示
                    self.visualizer.reset_aruco()
                    
                    # 在画面上显示未检测到标签
                    cv2.putText(canvas, "未检测到标签", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if not self.is_draw_camera:
                    # 绘制相机
                    self.visualizer.draw_camera()
                    self.is_draw_camera = True
                
                # 控制视野
                self.set_view_control()
                # 可视化器迭代
                self.visualizer.step()
                
                # 在画面上显示记录状态
                if self.recording:
                    cv2.putText(canvas, f"记录中... 已记录: {len(self.recorded_positions)}", 
                               (50, canvas.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("depth", depth_canvas)
                cv2.imshow("canvas", canvas)
                
                # 按键处理
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # 退出程序
                    break
                elif key == ord('r'):
                    # 开始/停止记录
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord('s'):
                    # 手动保存当前记录
                    self.save_positions()
        
        except Exception as e:
            print(f"运行中出错: {e}")
        finally:
            # 关闭窗口
            self.visualizer.destroy_window()
            # 释放相机
            self.camera.release()
            # 保存最终结果
            if self.recording and len(self.recorded_positions) > 0:
                self.save_positions()

if __name__ == "__main__":
    # 相机到世界坐标系的变换矩阵
    # 这里使用预先标定的矩阵，可以从外部文件加载或直接设置
    camera_to_world = np.array([
        [0.9995235405012892, 0.0026994478399852235, -0.03074743835064323, -28.244393511054284],
        [0.003988141025453063, -0.999112661826739, 0.04192831631069003, -701.4038878017706],
        [-0.030606971671965612, -0.04203096428643264, -0.9986473908874063, 845.7511974648796],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 输出文件路径
    output_file = "/home/xuan/dianrobot/wjx/eye/object_world_positions.txt"
    # 创建并运行物体跟踪器
    tracker = ObjectTracker(camera_to_world_matrix=camera_to_world, output_file=output_file)
    tracker.run()