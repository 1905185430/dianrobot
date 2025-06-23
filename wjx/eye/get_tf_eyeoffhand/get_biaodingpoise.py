'''
ArucoTag检测+位姿估计(3D相机版)
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
ros2
'''
import time
import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from kyle_robot_toolbox.camera import Gemini335
from kyle_robot_toolbox.opencv import ArucoTag
from kyle_robot_toolbox.open3d import *
from arucotag_visualizer import ArucoTagVisualizer
from arucotag_pose_adjust import *

class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_335')
        self.pub = self.create_publisher(String, 'tag_pose', 10)
        self.pub_list = self.create_publisher(Float32MultiArray, 'tag_trans_mat', 10)

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
        ctr = self.visualizer.visualizer.get_view_control()
        ctr.set_front(np.float64(self.view_point["front"]))
        ctr.set_lookat(np.float64(self.view_point["lookat"]))
        ctr.set_up(np.float64(self.view_point["up"]))
        ctr.set_zoom(np.float64(self.view_point["zoom"]))

    def run(self):
        while rclpy.ok():
            try:
                img_bgr, depth_img = self.camera.read()
                img_bgr = self.camera.remove_distortion(img_bgr)
                img_filter = image_preprocessor(img_bgr)
                depth_canvas = self.camera.depth_img2canvas(depth_img, min_distance=150, max_distance=300)
                scene_pcd = self.camera.get_pcd(img_bgr, depth_img)

                has_aruco, canvas, aruco_ids, aruco_centers, aruco_corners, T_cam2aruco_by2d = \
                    self.arucotag.aruco_pose_estimate(img_filter)

                self.visualizer.update_scene_pcd(scene_pcd)

                cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
                T_cam2aruco_by3d_filter = []

                if has_aruco:
                    valid_aruco_mask, t_cam2aruco_by3d_filter = get_t_cam2aruco_by3d( 
                        self.camera, depth_img, aruco_ids, aruco_centers, 
                        canvas=canvas, depth_canvas=depth_canvas)

                    aruco_ids_filter = aruco_ids[valid_aruco_mask]
                    aruco_centers_filter = aruco_centers[valid_aruco_mask]
                    aruco_corners_filter = aruco_corners[valid_aruco_mask]
                    T_cam2aruco_by2d_filter = T_cam2aruco_by2d[valid_aruco_mask]

                    T_cam2aruco_by3d_filter = adjust_T_cam2aruco(self.camera, img_filter, depth_img,
                        aruco_ids_filter, aruco_corners_filter,
                        T_cam2aruco_by2d_filter, t_cam2aruco_by3d_filter)

                    if len(T_cam2aruco_by3d_filter) > 0:
                        trans_matrix = T_cam2aruco_by3d_filter[0]
                        cam_x = trans_matrix[0,3]
                        cam_y = trans_matrix[1,3]
                        cam_z = trans_matrix[2,3]

                    self.visualizer.update_aruco(T_cam2aruco_by3d_filter)
                else:
                    self.visualizer.reset_aruco()

                if not self.is_draw_camera:
                    self.visualizer.draw_camera()
                    self.is_draw_camera = True

                self.set_view_control()
                self.visualizer.step()

                cv2.imshow("depth", depth_canvas)
                cv2.imshow("canvas", canvas)

                # 发布ROS2话题: 标签姿态(x,y,z)
                tag_pose_str = f"{cam_x} {cam_y} {cam_z}"
                msg = String()
                msg.data = tag_pose_str
                self.pub.publish(msg)

                # 发布ROS2话题: 变换矩阵
                mat_apart_data = Float32MultiArray()
                mat_apart = []
                print('T_cam2aruco_by3d_filter:', T_cam2aruco_by3d_filter)
                if len(T_cam2aruco_by3d_filter) > 0:
                    tag_trans_mat = [row.tolist() for row in T_cam2aruco_by3d_filter[0]]
                    for i in range(len(tag_trans_mat)):
                        for j in range(len(tag_trans_mat[i])):
                            mat_apart.append(tag_trans_mat[i][j])
                    mat_apart_data.data = mat_apart
                self.pub_list.publish(mat_apart_data)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            except Exception as e:
                print(e)
                self.visualizer.destroy_window()
                self.camera.release()
                break

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPosePublisher()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()