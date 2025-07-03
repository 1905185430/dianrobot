import time
import json
import numpy as np
import cv2
# import rospy
from std_msgs.msg import Float32MultiArray, String
# 阿凯机器人工具箱
# - Gemini335类
# from kyle_robot_toolbox.camera import Gemini335
# from kyle_robot_toolbox.opencv import ArucoTag
# from kyle_robot_toolbox.open3d import *
from fairino import Robot

#水平角度抓取
#rxryrz_0 = [90, 0, m]



class PathPlayer:
    """路径回放控制器，让机器人按照预先记录的路径点移动"""
    
    def __init__(self, robot, path_file=None):
        """
        初始化路径回放器
        
        参数:
            robot: 机器人控制对象
            path_file: 路径文件名，如果提供则自动加载
        """
        self.robot = robot
        self.waypoints = []  # 存储路径点 [x, y, z, rx, ry, rz]
        self.current_index = 0
        
        # 如果提供了文件路径，则加载路径
        if path_file:
            self.load_path_from_file(path_file)
    
    def load_path_from_file(self, filename):
        """
        从CSV文件加载路径点
        支持格式: CSV文件，表头为 X,Y,Z 或 X,Y,Z,RX,RY,RZ
        """
        import csv
        self.waypoints = []
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        z = float(row[2])
                        if len(row) >= 6:
                            rx = float(row[3])
                            ry = float(row[4])
                            rz = float(row[5])
                        else:
                            rx, ry, rz = 90.0, 0.0, 0.0  # 默认姿态
                        self.waypoints.append([x, y, z, rx, ry, rz])
                    except Exception:
                        continue
            print(f"成功加载 {len(self.waypoints)} 个路径点")
            print(f"路径点示例: {self.waypoints[:5]}")
        except Exception as e:
            print(f"加载路径文件错误: {e}")
    
    


    def play_path(self, velocity=20, blend_radius=0, pause_time=2):
        """
        按顺序执行路径中的所有点
        
        参数:
            velocity: 移动速度百分比 (0-100)
            blend_radius: 路径平滑半径(mm)，0表示精确经过每个点
            pause_time: 每个点之间的暂停时间(秒)
        
        返回:
            bool: 是否成功完成整个路径
        """
        if len(self.waypoints) == 0:
            print("路径为空，无法执行")
            return False
        robot_pose = self.robot.GetActualTCPPose(0)
            #       if robot_pose type is int
        while type(robot_pose) == int:
            robot_pose = self.robot.GetActualTCPPose(0)
            print("robot_error",robot_pose)
            time.sleep(0.5)
        print(f"开始执行路径，共 {len(self.waypoints)} 个点")
        input("按回车键开始执行路径...")
        self.waypoints = np.array(self.waypoints) + np.array([0, 135, -20, 0, 0, 0])  # 将所有点的姿态归一化为朝下的姿态
        robot.MoveGripper(1, 100, 50, 30, 10000, 1)
        start_pos0 = np.array(self.waypoints[0]) + np.array([0, 90, 150, 0, 0, 0])
        self.robot.MoveCart(start_pos0, 0, 0, vel=velocity, acc=20)
        input("按回车键继续...")
        start_pos = np.array(self.waypoints[0]) + np.array([0, 90, 0, 0, 0, 0])
        self.robot.MoveCart(start_pos, 0, 0, vel=velocity, acc=20)
        self.robot.MoveCart(self.waypoints[0], 0, 0, vel=velocity, acc=20)
        input("已到达初始位置，按回车键开始执行路径...")
        robot.MoveGripper(1, 70, 50, 30, 10000, 1)
        # 确保机器人准备就绪
        time.sleep(3)
        ret = self.robot.ServoMoveStart()
        if ret != 0:
            print(f"启动伺服模式失败，错误码: {ret}")
            return False
        
        success = True
        try:
            # 遍历所有路径点
            for i, point in enumerate(self.waypoints):
                #print(f"执行第 {i+1}/{len(self.waypoints)} 个点: {point}")
                
                
                ret = self.robot.ServoCart(0, point, cmdT = 0.008)
                #input(f"移动到点 {i+1}，按回车继续...")
        
                if ret != 0:
                    print(f"移动到点 {i+1} 失败，错误码: {ret}")
                    success = False
                    break
                
            
            print("路径执行完成" if success else "路径执行中断")
        
        except Exception as e:
            print(f"执行路径时发生错误: {e}")
            success = False
        
        finally:
            # 关闭伺服模式
            # self.robot.ServoMoveEnd()
            end1 = np.array(self.waypoints[-1])
            end1[3] = 90
            end1[4] = 0
            robot.MoveCart([0, -380, 200, 90, 0, 0], 0, 0, vel=20, acc=20)
            robot.MoveGripper(1, 100, 50, 30, 10000, 1)
            end_pos = np.array(self.waypoints[-1]) + np.array([0, 50, 0, 0, 0, 0])
            robot.MoveCart(end_pos, 0, 0, vel=20, acc=20)
        
        return success
    
    def play_path_2(self, velocity=20, blend_radius=0, pause_time=2):
            """
            按顺序执行路径中的所有点
            
            参数:
                velocity: 移动速度百分比 (0-100)
                blend_radius: 路径平滑半径(mm)，0表示精确经过每个点
                pause_time: 每个点之间的暂停时间(秒)
            
            返回:
                bool: 是否成功完成整个路径
            """
            if len(self.waypoints) == 0:
                print("路径为空，无法执行")
                return False
            robot_pose = self.robot.GetActualTCPPose(0)
                #       if robot_pose type is int
            while type(robot_pose) == int:
                robot_pose = self.robot.GetActualTCPPose(0)
                print("robot_error",robot_pose)
                time.sleep(0.5)
            print(f"开始执行路径，共 {len(self.waypoints)} 个点")
            input("按回车键开始执行路径...")
            # 加偏置
            self.waypoints = np.array(self.waypoints) + np.array([0, 135, 100, 0, 0, 0])  # 将所有点的姿态归一化为朝下的姿态
            # 打开夹爪
            robot.MoveGripper(1, 100, 50, 30, 10000, 1)
            # 机械臂移动到初始位置
            start_pos0 = np.array(self.waypoints[0]) + np.array([0, 90, 150, 0, 0, 0])
            out = self.robot.MoveCart(start_pos0, 0, 0, vel=velocity, acc=20)
            print("out:", out)
            input("按回车键继续...")
            start_pos = np.array(self.waypoints[0]) + np.array([0, 90, 0, 0, 0, 0])
            self.robot.MoveCart(start_pos, 0, 0, vel=velocity, acc=20)
            self.robot.MoveCart(self.waypoints[0], 0, 0, vel=velocity, acc=20)
            input("已到达初始位置，按回车键开始执行路径...")
            robot.MoveGripper(1, 70, 50, 30, 10000, 1)
            # 确保机器人准备就绪
            time.sleep(3)
            ret = self.robot.ServoMoveStart()
            if ret != 0:
                print(f"启动伺服模式失败，错误码: {ret}")
                return False
            
            success = True
            try:
                # 遍历所有路径点
                for i, point in enumerate(self.waypoints):
                    #print(f"执行第 {i+1}/{len(self.waypoints)} 个点: {point}")
                    if i != 0 and i % 200 == 0:
                        time.sleep(1)
                    time.sleep(0.001)  # 暂停一段时间
                    ret = self.robot.ServoCart(0, point, cmdT = 0.01)
                    #input(f"移动到点 {i+1}，按回车继续...")
            
                    if ret != 0:
                        print(f"移动到点 {i+1} 失败，错误码: {ret}")
                        success = False
                        break
                    
                
                print("路径执行完成" if success else "路径执行中断")
            
            except Exception as e:
                print(f"执行路径时发生错误: {e}")
                success = False
            
            finally:
                # 关闭伺服模式
                self.robot.ServoMoveEnd()
                robot.MoveGripper(1, 100, 50, 30, 10000, 1)
                end_pos = np.array(self.waypoints[-1]) + np.array([0, 50, 0, 0, 0, 0])
                robot.MoveCart(end_pos, 0, 0, vel=20, acc=20)
                robot.MoveGripper(1, 100, 50, 30, 10000, 1)
                end = np.array(self.waypoints[-1]) + np.array([0, 100, 0, 0, 0, 0])
                self.robot.MoveCart(end, 0, 0, vel=20, acc=20)
            return success

    
if __name__ == "__main__":
    robot = Robot.RPC('192.168.59.6')
    # robot = Robot.RPC('192.168.58.6')
    robot.ActGripper(1,0)
    time.sleep(1)
    robot.ActGripper(1,1)
    time.sleep(1)
    oo = robot.MoveCart([0, -380, 200, 90, 0, 0], 0, 0, vel=20, acc=20)
    print("", oo)
    time.sleep(5)
    PathPlayer0 = PathPlayer(robot, path_file='/home/xuan/dianrobot/wjx/eye/filled.csv')
    PathPlayer0.play_path_2(velocity=20, blend_radius=0, pause_time=0.008)
    time.sleep(2)
    # 机械臂移动到初始位置
    robot.MoveGripper(1, 100, 50, 30, 10000, 1)

    robot.MoveCart([0, -380, 200, 90, 0, 0], 0, 0, vel=20, acc=20)
    


