#!/bin/python3
# -*- coding: utf-8 -*-
'''
眼在手外 末端需要与标定板保持相对静止
可以得到相机相对于底座的变换矩阵，并自动保存
'''
import rospy
import os
import json
import time
import cv2
import numpy as np
from fairino import Robot
from std_msgs.msg import Float32MultiArray

# 使用说明：后面打包为roslaunch，使用时先开启get_biaodingpoise，在启动tf即可

# 最终目的，obj_to_base,需要obj_to_camera（视觉程序读）,camera_to_base（标定得到）
# 其中，camera_to_base需要标定得到，通过end_to_base（机械臂读取）,tag_to_end（未知，但固定）,camera_to_tag（视觉程序读）

# 将手眼标定方程（眼在手外）化为AX=XB形式：
# end2base_1*base2end_2 * camera2base = camera2base * tag2camera_1*camera2tag_2
# 标定利用opencv4中的calibrateHandEye()函数
# 传入7个参数，前四个是输入，然后是两个输出，最后是标定方法（默认tsai）

# camera2base = [ 
#  [ 0.99885901,  0.02414404  ,0.04120379,99.50256158],
#  [ 0.02214817, -0.99859083 , 0.04822672,-542.01673407],
#  [ 0.04231012 ,-0.0472591 , -0.99798619,846.15889269],
#  [ 0.0,0.0,0.0,1.0]
# ]
camera2base = [[0.032418495539366265, -0.24541720737592693, 0.9688753456821808, -1034.0775401017188], 
               [-0.9994715684695943, -0.010260518882398317, 0.03084324193714344, -323.55591698774987], 
               [0.0023717014764284006, -0.9693632529016485, -0.2456201517710661, 26.88006545580754], 
               [0.0, 0.0, 0.0, 1.0]]
tag_trans_mat = []
fr5_A = []
robot = Robot.RPC('192.168.59.6')

def init():
    '''
    初始化函数，包含机械臂初始化
    '''
    global fr5_A
    fr5_A = robot
    time.sleep(0.5)
# 补充机械臂初始化的函数代码

def save_to_file(matrix):
    # 创建log文件夹（如果不存在）
    log_dir = "/home/hn/One-shot-imitation-in-chemistry-lab-main/eye/biaoding"
    file_name = "camera2base_cjj.txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 确定文件路径和名称
    file_path = os.path.join(log_dir, file_name)

    # 如果文件已经存在，则找一个可用的文件名
    index = 1
    while os.path.exists(file_path):
        index += 1
        file_path = os.path.join(log_dir, f"camera2base_cjj{index}.txt")
        print('repeat file name')

    # 将矩阵转换为Python列表
    matrix_list = matrix.tolist()

    # 保存矩阵列表到文件
    with open(file_path, 'w') as file:
        # 序列化矩阵列表为JSON字符串
        matrix_str = json.dumps(matrix_list)
        file.write(matrix_str)
        print('file saved')

def camera_callback2(rece_tag_trans_mat):
    '''
    回调函数，得到tag_to_camera的变换矩阵
    由于ros功能限制，在此将二维数组压缩为一维数组接收，需要做对应解码处理
    @输入：
        rece_tag_trans_mat：ros发来的tag2camera信息
    @输出：
        None
    '''
    global tag_trans_mat
    #rospy.loginfo("自发自收收到的tag_trans_mat数据: %s" % str(rece_tag_trans_mat.data))
    # print("Received tag_trans_mat:\n", rece_tag_trans_mat.data, '\n')
    # exit()
    if rece_tag_trans_mat.data == []:
        pass
    else :
        tag_trans_mat = rece_tag_trans_mat.data
        tag_trans_mat = list(tag_trans_mat)
        # 将一维数组重塑为4x4矩阵
        tag_trans_mat = [tag_trans_mat[i:i+4] for i in range(0, len(tag_trans_mat), 4)]
        # print(tag_trans_mat,'\n')

# def camera_callback2(rece_tag_trans_mat):
#     '''
#     回调函数，得到tag_to_camera的变换矩阵
#     由于ros功能限制，在此将二维数组压缩为一维数组接收，需要做对应解码处理
#     @输入：
#         rece_tag_trans_mat：ros发来的tag2camera信息 (Float32MultiArray类型)
#     @输出：
#         None
#     '''
#     global tag_trans_mat
#     data = rece_tag_trans_mat.data
#     if len(data) == 16:
#         # 将一维数组重塑为4x4矩阵
#         tag_trans_mat = np.array(data).reshape(4,4)
#         print("Received 4x4 tag_trans_mat:\n", tag_trans_mat, '\n')
#     else:
#         # 数据为空或者长度不匹配时输出提示
#         print("Received empty or invalid tag_trans_mat data:\n", data, '\n')
#         tag_trans_mat = np.array([])  # 根据需求设置为空或默认值


def get_transform_mat(X,Y,Z,RX,RY,RZ):
    '''
    从机械臂末端6D数据得到end2base变换矩阵
    @输入：
        XYZ,RXRYRZ：机械臂末端6D数据
    @输出：
        end_to_base：机械臂end2base数据
    '''
    # 旋转角度
    rx = np.deg2rad(RX)
    ry = np.deg2rad(RY)
    rz = np.deg2rad(RZ)

    # 绕x轴旋转矩阵
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])

    # 绕y轴旋转矩阵
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]])

    # 绕z轴旋转矩阵
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]])

    # 旋转矩阵的乘积
    R = np.dot(np.dot(Rz, Ry),Rx)

    # 平移向量
    tx = X
    ty = Y
    tz = Z

    # 变换矩阵
    end_to_base = np.array([[R[0, 0], R[0, 1], R[0, 2], tx],
                [R[1, 0], R[1, 1], R[1, 2], ty],
                [R[2, 0], R[2, 1], R[2, 2], tz],
                [0, 0, 0, 1]])
    return end_to_base
    
def tf_get_obj_to_base(obj2camera,camera2base):
    '''
    得到obj2base变换矩阵
    @输入：
        obj2camera：物体在相机坐标系下的变换矩阵
        camera2base：相机在机械臂底座下的变换矩阵
    @输出：
        obj2base：物体在机械臂底座下的变换矩阵
    '''
    obj2base = np.dot(camera2base,obj2camera)
    return obj2base
    
def get_RT_from_transform_mat(transform_mat):
    '''
    给定变换矩阵，给出旋转矩阵与平移向量
    @输入：
        transform_mat：待拆解的变换矩阵
    @输出：
        rot_mat：旋转矩阵
        translate_mat:平移向量
    '''
    rot_mat = transform_mat[:3,:3]
    translate_mat = transform_mat[:3,3]
    return rot_mat,translate_mat

def get_transform_mat_from_RT(R,T):
    '''
    给定旋转矩阵和平移向量，给出变换矩阵
    @输入：
        R：旋转矩阵
        T：平移向量
    @输出：
        M：变换矩阵
    '''
    Z = [0.0,0.0,0.0,1.0]
    M = np.vstack((np.hstack((R, T)), Z))
    return M



def main():
    
    #旋转​​（R）：3×3矩阵，描述目标坐标系相对于参考坐标系的旋转姿态。
    #平移（T）：3×1向量，描述目标坐标系相对于参考坐标系的平移位置。

#   ​前3×3子矩阵​​：旋转矩阵 R。
# ​  ​第4列前3行​​：平移向量 T。
# ​  ​最后一行​​：齐次坐标补位（固定为 [0, 0, 0, 1]）
    R_base2end_list = []
    T_base2end_list = []
    R_tag2camera_list = [] 
    T_tag2camera_list = []
    R_camera2base = []
    T_camera2base = []
    R_camera2base = np.array(R_camera2base)
    T_camera2base = np.array(T_camera2base)
    global tag_trans_mat
    rospy.init_node('tag_trans_mat_listener', anonymous=True)
    rospy.Subscriber('/tag_trans_mat',Float32MultiArray,camera_callback2)
    sample_times = input('------请输入采集次数------')
    input('------等待按下回车开始采集数据------')
    for i in range(int(sample_times)):
        #获取机械臂末端执行器的实际位姿
        fr5_A_end = fr5_A.robot.GetActualToolFlangePose(0)
        fr5_A_end = fr5_A_end[-6:]# 得到机械臂末端xyz，rxryrz

        # 得到end2base矩阵
        end2base = get_transform_mat(fr5_A_end[0],fr5_A_end[1],fr5_A_end[2],fr5_A_end[3],fr5_A_end[4],fr5_A_end[5])
        # print('end2base:',end2base)

        # 得到并处理base2end矩阵（求逆矩阵）
        base2end = np.linalg.inv([end2base])
        base2end = base2end[0]
        print('base2end:',base2end)

        # 得到base2end的旋转矩阵与平移向量
        R_base2end , T_base2end = get_RT_from_transform_mat(base2end)

        # 得到tag2camera的旋转矩阵与平移向量
        print('tag_trans_mat',tag_trans_mat)
        tag_trans_mat = np.array(tag_trans_mat)
        R_tag2camera , T_tag2camera = get_RT_from_transform_mat(tag_trans_mat)

        # 把上述四个矩阵制成列表
        R_base2end_list.append(R_base2end)
        T_base2end_list.append(T_base2end)
        R_tag2camera_list.append(R_tag2camera)
        T_tag2camera_list.append(T_tag2camera)

        input('--------等待调整末端姿态并重新记录--------')

    # 创建一个字典，用于存储矩阵和对应的文字说明
    matrix_dict = {
        "R_base2end_list": R_base2end_list,
        "T_base2end_list": T_base2end_list,
        "R_tag2camera_list": R_tag2camera_list,
        "T_tag2camera_list": T_tag2camera_list
    }

    #得到camera2base的旋转矩阵与平移向量
    R_camera2base,T_camera2base = cv2.calibrateHandEye(R_base2end_list,T_base2end_list,R_tag2camera_list,T_tag2camera_list,method=cv2.CALIB_HAND_EYE_TSAI)

    print('R_camera2base',R_camera2base)
    print('T_camera2base',T_camera2base)
    # 保存变换矩阵
    save_to_file(get_transform_mat_from_RT(R_camera2base,T_camera2base))
    # # 打印矩阵和文字说明
    # for key, matrix in matrix_dict.items():
    #     print(key + ":")
    #     if isinstance(matrix, np.ndarray):
    #         matrix = matrix.tolist()
    #     print(matrix)
    #     print("\n")

def test():
    global tag_trans_mat
    #初始化ROS节点：创建名为'fr5_main'的ROS节点，anonymous=True确保节点名称唯一
    rospy.init_node('fr5_main', anonymous=True)
    #订阅ROS主题：订阅名为/tag_trans_mat的主题，接收类型为Float32MultiArray的消息
    #当主题/tag_trans_mat发布新消息时，camera_callback2被触发
    rospy.Subscriber('/tag_trans_mat',Float32MultiArray,camera_callback2)
    while True:
        input('等待按下回车进行一次计算：')
        # print('tag2camera', tag_trans_mat)
        # print('cameara2base', camera2base)
        
        # tag_trans_mat 作用：标定板在相机坐标系下的位姿
        obj2base = tf_get_obj_to_base(tag_trans_mat,camera2base)
        print('结果为：',obj2base)

def newtest():
    # R = [[ 0.99885901 , 0.02414404 , 0.04120379],
    #     [ 0.02214817 ,-0.99859083 , 0.04822672],
    #     [ 0.04231012 ,-0.0472591  ,-0.99798619]]
    # T = [[99.50256158],
    #      [ -542.01673407],
    #      [ 846.15889269]]
    R = [[-0.99834368,  0.00331012, -0.05743647],
        [ 0.00833953, 0.99612544, -0.08754746],
        [ 0.05692413, -0.08788145, -0.99450314]]
    T = [[ -75.16659019],
        [-771.97540218],
        [ 857.6738249 ]]
    print(get_transform_mat_from_RT(R,T))
    save_to_file(get_transform_mat_from_RT(R,T))

if __name__ == "__main__":
    try:
        init()
        # 标定模式：采集数据并计算相机到机器人基座的变换矩阵
        main()
        # 测试模式：使用已有的变换矩阵测试坐标转换精度
        # test()
    except rospy.ROSInterruptException as e:
        print(e,"\n")