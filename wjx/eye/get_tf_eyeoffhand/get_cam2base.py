#!/bin/python3
# -*- coding: utf-8 -*-
'''
眼在手外 末端需要与标定板保持相对静止
可以得到相机相对于底座的变换矩阵，并自动保存
'''
import rclpy
from rclpy.node import Node
import os
import json
import time
import cv2
import numpy as np
from fairino import Robot
from std_msgs.msg import Float32MultiArray
import random

error_codes = {
    -7: ("上传文件不存在", "检查文件名称是否正确"),
    -6: ("保存文件路径不存在", "检查文件路径是否正确"),
    -5: ("LUA文件不存在", "检查lua文件名称是否正确"),
    -4: ("xmlrpc接口执行失败", "请联系后台工程"),
    -3: ("xmlrpc通讯失败", "请检查网络连接及服务器IP地址是否正确"),
    -2: ("与控制器通讯异常", "检查与控制器硬件连接"),
    -1: ("其他错误", "联系售后工程师查看控制器日志"),
    0: ("调用成功", ""),
    1: ("接口参数个数不一致", "检查接口参数个数"),
    3: ("接口参数值异常", "检查参数类型或范围"),
    8: ("轨迹文件打开失败", "检查TPD轨迹文件是否存在或轨迹名是否正确"),
    9: ("TPD文件名发送失败", "检查TPD轨迹名是否正确"),
    10: ("TPD文件内容发送失败", "检查TPD文件内容是否正确")
}

camera2base=[[-0.9901652346580831, 0.013433445840222654, 0.13925642034520688, -27.163789618795875], 
             [0.006592239752229778, 0.9987537537350062, -0.049472030232080154, -593.2150728299232], 
             [-0.13974745239020378, -0.048067472713806535, -0.9890198014283411, 834.0095420825568], 
             [0.0, 0.0, 0.0, 1.0]]
tag_trans_mat = []
fr5_A = []
robot = Robot.RPC('192.168.58.6')  # 替换为实际的机器人IP地址

def get_random_xyz_pos():
    x = random.uniform(-300, 350)
    y = random.uniform(-680, -350)
    z = random.uniform(200, 350)    
    rx = random.uniform(80, 120)
    ry = random.uniform(-30, 30)
    rz = random.uniform(-30, 30)
    return [x, y, z, rx, ry, rz]

def move_to_pose(robot, target_position):
    ret = robot.MoveCart(target_position, 0, 0)
    if ret != 0:
        print(f"机械臂运动失败，错误码: {ret}")
        raise RuntimeError(f"机械臂运动失败，错误码: {ret}")
    print(f"机械臂已移动到目标位姿: {target_position}")
    time.sleep(1.5)

def init():
    global fr5_A
    fr5_A = robot
    robot.ActGripper(1,0)
    time.sleep(1)
    robot.ActGripper(1,1)
    time.sleep(1)
    robot.MoveGripper(1, 100, 50, 30, 10000, 1)
    input('------等待按下回车夹住------')
    robot.MoveGripper(1, 72, 50, 30, 10000, 1)
    time.sleep(0.5)

def save_to_file(matrix):
    log_dir = "/home/xuan/dianrobot/wjx/eye/biaoding"
    file_name = "camera2base.txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = os.path.join(log_dir, file_name)
    index = 1
    while os.path.exists(file_path):
        index += 1
        file_path = os.path.join(log_dir, f"camera2base_cjj{index}.txt")
        print('repeat file name')
    matrix_list = matrix.tolist()
    with open(file_path, 'w') as file:
        matrix_str = json.dumps(matrix_list)
        file.write(matrix_str)
        print('file saved')

def get_transform_mat(X,Y,Z,RX,RY,RZ):
    rx = np.deg2rad(RX)
    ry = np.deg2rad(RY)
    rz = np.deg2rad(RZ)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry),Rx)
    tx = X
    ty = Y
    tz = Z
    end_to_base = np.array([[R[0, 0], R[0, 1], R[0, 2], tx],
                [R[1, 0], R[1, 1], R[1, 2], ty],
                [R[2, 0], R[2, 1], R[2, 2], tz],
                [0, 0, 0, 1]])
    return end_to_base

def tf_get_obj_to_base(obj2camera,camera2base):
    obj2base = np.dot(camera2base,obj2camera)
    return obj2base

def get_RT_from_transform_mat(transform_mat):
    rot_mat = transform_mat[:3,:3]
    translate_mat = transform_mat[:3,3]
    return rot_mat,translate_mat

def get_transform_mat_from_RT(R,T):
    Z = [0.0,0.0,0.0,1.0]
    M = np.vstack((np.hstack((R, T)), Z))
    return M

class TagTransMatListener(Node):
    def __init__(self):
        super().__init__('tag_trans_mat_listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/tag_trans_mat',
            self.camera_callback2,
            10)
        self.tag_trans_mat = []

    def camera_callback2(self, rece_tag_trans_mat):
        global tag_trans_mat
        if len(rece_tag_trans_mat.data) == 0:
            return
        tag_trans_mat = list(rece_tag_trans_mat.data)
        tag_trans_mat = [tag_trans_mat[i:i+4] for i in range(0, len(tag_trans_mat), 4)]
        self.tag_trans_mat = tag_trans_mat

def main(args=None):
    global tag_trans_mat
    rclpy.init(args=args)
    listener = TagTransMatListener()
    sample_times = int(input('------请输入采集次数------'))
    input('------等待按下回车开始采集数据------')
    R_base2end_list = []
    T_base2end_list = []
    R_tag2camera_list = [] 
    T_tag2camera_list = []
    for i in range(sample_times):
        move_to_pose(robot, get_random_xyz_pos())
        time.sleep(0.5)
        fr5_A_end = fr5_A.robot.GetActualToolFlangePose(0)
        fr5_A_end = fr5_A_end[-6:]
        end2base = get_transform_mat(fr5_A_end[0],fr5_A_end[1],fr5_A_end[2],fr5_A_end[3],fr5_A_end[4],fr5_A_end[5])
        base2end = np.linalg.inv([end2base])[0]
        print('base2end:',base2end)
        R_base2end , T_base2end = get_RT_from_transform_mat(base2end)
        # 等待tag_trans_mat更新
        for _ in range(100):
            rclpy.spin_once(listener, timeout_sec=0.1)
            if hasattr(listener, 'tag_trans_mat') and len(listener.tag_trans_mat) == 4:
                break
        print('tag_trans_mat', listener.tag_trans_mat)
        if not hasattr(listener, 'tag_trans_mat') or len(listener.tag_trans_mat) == 0:
            continue
        tag_mat_np = np.array(listener.tag_trans_mat)
        R_tag2camera , T_tag2camera = get_RT_from_transform_mat(tag_mat_np)
        R_base2end_list.append(R_base2end)
        T_base2end_list.append(T_base2end)
        R_tag2camera_list.append(R_tag2camera)
        T_tag2camera_list.append(T_tag2camera)
    R_camera2base,T_camera2base = cv2.calibrateHandEye(
        R_base2end_list,T_base2end_list,R_tag2camera_list,T_tag2camera_list,method=cv2.CALIB_HAND_EYE_TSAI)
    print('R_camera2base',R_camera2base)
    print('T_camera2base',T_camera2base)
    save_to_file(get_transform_mat_from_RT(R_camera2base,T_camera2base))
    time.sleep(1)
    robot.MoveCart([0, -400, 200, 90, 0, 0], 0, 0, vel=20, acc=20)
    time.sleep(1)
    input('------等待按下回车夹住标定板------')
    robot.MoveGripper(1, 100, 50, 30, 10000, 1)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    try:
        init()
        main()
    except KeyboardInterrupt as e:
        print(e,"\n")