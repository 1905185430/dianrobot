'''
手掌关键点检测与定位
- 获取手掌0号点在相机坐标系下的三维坐标
- 支持记录手部运动轨迹
'''
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
import os
import time
from datetime import datetime
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335

class HandDetector:
    def __init__(self):
        # 创建相机对象
        self.camera = Gemini335()
        # 获取相机内参
        self.intrinsic = self.camera.get_intrinsic()
        # 59
        # self.camera_to_world = np.array([
        #     [0.9995235405012892, 0.0026994478399852235, -0.03074743835064323, -28.244393511054284],
        #     [0.003988141025453063, -0.999112661826739, 0.04192831631069003, -701.4038878017706],
        #     [-0.030606971671965612, -0.04203096428643264, -0.9986473908874063, 845.7511974648796],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])
        # 58
        self.camera_to_world = [[-0.9901652346580831, 0.013433445840222654, 0.13925642034520688, -27.163789618795875], 
                                [0.006592239752229778, 0.9987537537350062, -0.049472030232080154, -593.2150728299232], 
                                [-0.13974745239020378, -0.048067472713806535, -0.9890198014283411, 834.0095420825568], 
                                [0.0, 0.0, 0.0, 1.0]]
        # 获取彩图
        self.color_img = None
        # 获取深度图
        self.depth_img = None
        # 生成画布
        # min_detection_confidence: 置信度阈值 
        # min_tracking_confidence: 跟踪阈值
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3)
        # 绘图
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        # 检测结果
        self.results = None
        self.obj_pos_3d = None
        self.obj_pos_2d = None
        self.dot_0_pos = None
        self.dot_4_pos = None
        self.dot_8_pos = None

        # 通过048点计算二维抓取角度
        self.rx = None
        self.ry = None
        self.rz = None

        # 计算抓取角度依据点1234
        self.dot_1_pos = None
        self.dot_2_pos = None
        self.dot_3_pos = None
        self.dot_4_pos = None
        self.direction = None  # 主方向向量
        self.rxryrz = None  # 抓取角度 (rx, ry, rz)

        # 记录手部关键点的世界坐标
        self.obs_pos = None

    def vector_to_euler(self):
        """
        # 待修
        将主方向向量转为欧拉角(rx, ry, rz)，ry默认为0
        返回单位为度
        """
        if self.direction is None:
            print("[WARN] 主方向向量未计算，无法转换为欧拉角")
            self.rx, self.ry, self.rz = None, None, None
            return None, None, None
        v = self.direction
        # 归一化
        v = v / np.linalg.norm(v)

        reference=np.array([0, 0, 1])
        reference = reference / np.linalg.norm(reference)
        # 计算旋转轴和角度
        axis = np.cross(reference, v)
        angle = np.arccos(np.clip(np.dot(reference, v), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            rot_matrix = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            rvec = axis * angle
            rot_matrix, _ = cv2.Rodrigues(rvec)
        # 用OpenCV分解欧拉角（XYZ顺序）
        euler_angles, _, _, _, _, _  = cv2.RQDecomp3x3(rot_matrix)
        rx, ry, rz = euler_angles
        print(f"[INFO] 主方向向量: {self.direction}, 欧拉角: rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")
        return rx, ry, rz

    def transform_to_world_coordinates(self, point3d):
        """
        将相机坐标系中的点转换到世界坐标系
        
        参数:
        point3d: 相机坐标系下的三维点 (x, y, z)
        
        返回:
        世界坐标系下的三维点 (x, y, z)
        """

        point3d_homogeneous = np.append(point3d, 1.0)
        world_point = np.dot(self.camera_to_world, point3d_homogeneous)
        return world_point[:3]
    

    def calculate_obj_position(self, roi_size=5):
        """
        计算食指指尖与拇指指尖的3D坐标，并返回它们中间点在世界坐标系下的三维坐标

        返回:
        [world_x, world_y, world_z], px, py
        world_x, world_y, world_z: 世界坐标系下的三维坐标（中点）
        px, py: 食指指尖与拇指指尖中间点的像素坐标
        roi_size: 深度图ROI区域大小，默认5像素
        """
        if self.results.multi_hand_landmarks is not None:
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            # 获取Landmark的像素坐标
            px_4 = int(pixel_landmark[4].x * self.camera.img_width)
            py_4 = int(pixel_landmark[4].y * self.camera.img_height)
            px_8 = int(pixel_landmark[8].x * self.camera.img_width)
            py_8 = int(pixel_landmark[8].y * self.camera.img_height)
            px = int((px_4 + px_8) / 2)
            py = int((py_4 + py_8) / 2)

            # 获取深度图
            depth_img = self.depth_img
            if depth_img is None:
                print("[WARN] 深度数据帧获取失败，跳过当前帧处理")
                return None, None

            dp_h, dp_w = depth_img.shape

            # 获取拇指指尖ROI
            y_min_4 = max(0, py_4 - roi_size)
            y_max_4 = min(dp_h, py_4 + roi_size)
            x_min_4 = max(0, px_4 - roi_size)
            x_max_4 = min(dp_w, px_4 + roi_size)
            roi_4 = depth_img[y_min_4:y_max_4, x_min_4:x_max_4]
            index_4 = np.where(roi_4 != 0)

            # 获取食指指尖ROI
            y_min_8 = max(0, py_8 - roi_size)
            y_max_8 = min(dp_h, py_8 + roi_size)
            x_min_8 = max(0, px_8 - roi_size)
            x_max_8 = min(dp_w, px_8 + roi_size)
            roi_8 = depth_img[y_min_8:y_max_8, x_min_8:x_max_8]
            index_8 = np.where(roi_8 != 0)

            if len(index_4[0]) > 0 and len(index_8[0]) > 0:
                depth_value_4 = np.mean(roi_4[index_4])
                depth_value_8 = np.mean(roi_8[index_8])
                # 计算3D坐标
                cam_point3d_4 = self.camera.depth_pixel2cam_point3d(
                    px_4, py_4, depth_value=depth_value_4, intrinsic=self.intrinsic)
                cam_point3d_8 = self.camera.depth_pixel2cam_point3d(
                    px_8, py_8, depth_value=depth_value_8, intrinsic=self.intrinsic)
                # 转世界坐标
                world_point_4 = self.transform_to_world_coordinates(cam_point3d_4)
                world_point_8 = self.transform_to_world_coordinates(cam_point3d_8)
                # 求中点
                world_point = (np.array(world_point_4) + np.array(world_point_8)) / 2.0
                return world_point.tolist(), [px, py]
            else:
                print("[WARN] ROI 区域无有效深度值")
                return None, None
        else:
            print("[INFO] 未检测到手部关键点")
            return None, None

    def get_pos_2d(self, dot_num=0):
        """
        获取手部关键点的像素坐标
        dot_num: 0-20号点
        返回: (px, py)
        """
        if self.results.multi_hand_landmarks is not None:
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            px = int(pixel_landmark[dot_num].x * self.camera.img_width)
            py = int(pixel_landmark[dot_num].y * self.camera.img_height)
            return px, py
        else:
            print("[INFO] 未检测到手部关键点")
            return None, None
        
    def get_dot_pos_3d(self, dot_num=0, roi_size=5):
        '''
        获取手部关键点的世界坐标
        dot_num: 0-20号点
        roi_size: 深度图ROI区域大小，默认5像素
        返回: (world_x, world_y, world_z)
        world_x, world_y, world_z: 世界坐标系下的三维坐标
        '''
        if self.results.multi_hand_landmarks is not None:
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            # 获取关键点的像素坐标
            px, py = self.get_pos_2d(dot_num=dot_num)

            # 获取深度图
            depth_img = self.depth_img
            if depth_img is None:
                print("[WARN] 深度数据帧获取失败，跳过当前帧处理")
                return None

            dp_h, dp_w = depth_img.shape

            # 根据像素坐标获取深度值
            y_min = max(0, py - roi_size)
            y_max = min(dp_h, py + roi_size)
            x_min = max(0, px - roi_size)
            x_max = min(dp_w, px + roi_size)

            # 获取ROI区域
            roi = depth_img[y_min:y_max, x_min:x_max]

            # 过滤无效深度值
            index = np.where(roi != 0)
            if len(index[0]) > 0:
                depth_value = np.mean(roi[index])
                # 将像素坐标转换为相机坐标系下的三维坐标
                cam_point3d = self.camera.depth_pixel2cam_point3d(
                    px, py, depth_value=depth_value, intrinsic=self.intrinsic)
                cam_x, cam_y, cam_z = cam_point3d
                # 转换到世界坐标系
                world_point = self.transform_to_world_coordinates([cam_x, cam_y, cam_z])
                return world_point
            else:
                print("[WARN] ROI 区域无有效深度值")
                return None
        else:
            print("[INFO] 未检测到手部关键点")
            return None
        

    def get_048_pos(self, roi_size=5):
        """
        获取0、4、8号点的世界坐标
        roi_size: 深度图ROI区域大小，默认5像素
        返回: (world_point_0, world_point_4, world_point_8)
        world_point_0, world_point_4, world_point_8: 世界坐标系下的三维坐标
        0号点: 手腕
        4号点: 拇指指尖
        8号点: 食指指尖
        """

        if self.results.multi_hand_landmarks is not None:
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            # 获取关键点0,4,8的像素坐标
            px_0, py_0 = self.get_pos_2d(dot_num=0)
            px_4, py_4 = self.get_pos_2d(dot_num=4)
            px_8, py_8 = self.get_pos_2d(dot_num=8)

            # 获取深度图
            depth_img = self.depth_img
            if depth_img is None:
                print("[WARN] 深度数据帧获取失败，跳过当前帧处理")
                return None

            dp_h, dp_w = depth_img.shape

            # 根据像素坐标获取深度值
            y_min_0 = max(0, py_0 - roi_size)
            y_max_0 = min(dp_h, py_0 + roi_size)
            x_min_0 = max(0, px_0 - roi_size)
            x_max_0 = min(dp_w, px_0 + roi_size)

            y_min_4 = max(0, py_4 - roi_size)
            y_max_4 = min(dp_h, py_4 + roi_size)
            x_min_4 = max(0, px_4 - roi_size)
            x_max_4 = min(dp_w, px_4 + roi_size)

            y_min_8 = max(0, py_8 - roi_size)
            y_max_8 = min(dp_h, py_8 + roi_size)
            x_min_8 = max(0, px_8 - roi_size)
            x_max_8 = min(dp_w, px_8 + roi_size)

            # 获取ROI区域
            roi_0 = depth_img[y_min_0:y_max_0, x_min_0:x_max_0]
            roi_4 = depth_img[y_min_4:y_max_4, x_min_4:x_max_4]
            roi_8 = depth_img[y_min_8:y_max_8, x_min_8:x_max_8]
            # 过滤无效深度值
            index_0 = np.where(roi_0 != 0)
            index_4 = np.where(roi_4 != 0)
            index_8 = np.where(roi_8 != 0)
            if len(index_0[0]) > 0 and len(index_4[0]) > 0 and len(index_8[0]) > 0:  # 确保有有效数据点
                depth_value_0 = np.mean(roi_0[index_0])  # 计算有效深度值的平均值
                depth_value_4 = np.mean(roi_4[index_4])
                depth_value_8 = np.mean(roi_8[index_8])
                # 将像素坐标转换为相机坐标系下的三维坐标
                cam_point3d_0 = self.camera.depth_pixel2cam_point3d(
                    px_0, py_0, depth_value=depth_value_0, intrinsic=self.intrinsic)
                cam_point3d_4 = self.camera.depth_pixel2cam_point3d(
                    px_4, py_4, depth_value=depth_value_4, intrinsic=self.intrinsic)
                cam_point3d_8 = self.camera.depth_pixel2cam_point3d(
                    px_8, py_8, depth_value=depth_value_8, intrinsic=self.intrinsic)
                # 转换到世界坐标系
                world_point_0 = self.transform_to_world_coordinates(cam_point3d_0)
                world_point_4 = self.transform_to_world_coordinates(cam_point3d_4)
                world_point_8 = self.transform_to_world_coordinates(cam_point3d_8)
                return world_point_0, world_point_4, world_point_8
            else:
                print("[WARN] ROI 区域无有效深度值")
                return None, None, None
        else:
            print("[INFO] 未检测到手部关键点")
            return None, None, None

    def fit_direction_vector(self):
        """
        用最小二乘法拟合空间点的主方向向量
        返回: 单位化的主方向向量 direction (dx, dy, dz)
        向量的方向与最后一个点减去第一个点的方向一致
        """
        # 过滤掉None点
        points = [p for p in [self.dot_1_pos, self.dot_2_pos, self.dot_3_pos, self.dot_4_pos] if p is not None]
        if len(points) < 2:
            print("[WARN] 有效点数不足，无法拟合主方向")
            self.direction = None
            return None
        points = np.array(points)
        # 去中心化
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        # 计算协方差矩阵
        cov = np.cov(centered, rowvar=False)
        # 求最大特征值对应的特征向量
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        # 单位化
        direction = direction / np.linalg.norm(direction)
        # 保证方向与最后一个点减第一个点一致
        ref_vec = points[-1] - points[0]
        if np.dot(direction, ref_vec) < 0:
            direction = -direction
        self.direction = direction
        return direction

    def get_obs(self):
        """
        获取观察点位置
        返回: 观察点位置 obs_pos (world_x, world_y, world_z, rx, ry, rz)
        """
        if self.obj_pos_3d is None:
            print("[WARN] obj_pos_3d is None, 无法获取观察点位置")
            self.obs_pos = None
            return None
        if self.rx is None or self.ry is None or self.rz is None:
            print("[WARN] 抓取角度未计算，无法获取观察点位置")
            self.obs_pos = None
            return None
        # 计算观察点位置
        if isinstance(self.obj_pos_3d, np.ndarray):
            self.obs_pos = np.concatenate([self.obj_pos_3d, np.array([self.rx, self.ry, self.rz])]).tolist()
        else:
            self.obs_pos = list(self.obj_pos_3d) + [self.rx, self.ry, self.rz]
        print(f"[INFO] 观察点位置: {self.obs_pos}")

    def update(self):
        """
        更新手部关键点检测图像
        更新手部关键点检测结果
        更新手部关键点的世界坐标
        获取0、4、8号点的世界坐标
        计算抓取角度
        """
        self.depth_img = self.camera.read_depth_img()
        self.color_img = cv2.cvtColor(self.camera.read_color_img(), cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.color_img)
        # 更新手部关键点的世界坐标
        self.obj_pos_3d, self.obj_pos_2d = self.calculate_obj_position(roi_size=5)
        # 获取0、4、8号点的世界坐标
        self.dot_0_pos, self.dot_4_pos, self.dot_8_pos = self.get_048_pos(roi_size=5)
        # 计算1234号点的世界坐标
        self.dot_1_pos = self.get_dot_pos_3d(dot_num=1, roi_size=5)
        self.dot_2_pos = self.get_dot_pos_3d(dot_num=2, roi_size=5)
        self.dot_3_pos = self.get_dot_pos_3d(dot_num=3, roi_size=5) 
        self.dot_4_pos = self.get_dot_pos_3d(dot_num=4, roi_size=5)
        # 计算主方向向量
        self.fit_direction_vector()
        # 计算抓取角度
        self.vector_to_euler()
        print(f"[INFO] 抓取角度: rx={self.rx}°, ry={self.ry}°, rz={self.rz}°")
        self.get_obs()

    def save_depth_rgb_and_pointcloud(self, filename_prefix="capture"):
        """
        获取当前深度图和彩色图，并保存到当前目录，同时生成点云数据保存为ply文件
        文件名格式: capture_depth.png, capture_rgb.png, capture_pointcloud.ply
        """
        # 获取深度图和彩色图
        depth_img = self.camera.read_depth_img()
        color_img = self.camera.read_color_img()

        # 保存深度图
        depth_path = f"{filename_prefix}_depth.png"
        cv2.imwrite(depth_path, depth_img)
        print(f"[INFO] 深度图已保存到 {depth_path}")

        # 保存RGB图
        rgb_path = f"{filename_prefix}_rgb.png"
        cv2.imwrite(rgb_path, color_img)
        print(f"[INFO] RGB图已保存到 {rgb_path}")

        # 生成点云
        # 获取相机内参
        intrinsic = self.intrinsic
        h, w = depth_img.shape
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                d = depth_img[v, u]
                if d == 0:
                    continue
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                b, g, r = color_img[v, u] if color_img.ndim == 3 else (0, 0, 0)
                colors.append([r / 255.0, g / 255.0, b / 255.0])

        if len(points) == 0:
            print("[WARN] 没有有效点云数据")
            return

        # 保存点云为ply
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.array(points))
        pc.colors = o3d.utility.Vector3dVector(np.array(colors))
        ply_path = f"{filename_prefix}_pointcloud.ply"
        o3d.io.write_point_cloud(ply_path, pc)
        print(f"[INFO] 点云已保存到 {ply_path}")

    
    def camera_exit(self):
        self.camera.release()
        print("释放相机资源")


if __name__ == "__main__":
    detector = HandDetector()
    input("按回车键开始手部检测...")
    detector.update()
    detector.save_depth_rgb_and_pointcloud(filename_prefix="hand_capture") 
    detector.camera_exit()
    print("手部检测完成，退出程序")                  
        