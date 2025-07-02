'''
输入：手的rgb图片  用于得到食指和拇指的mask
     深度图       用于计算食指和手指形成的平面
'''

import cv2
import mediapipe as mp
import numpy as np
import SpaceMath

# 初始化 Mediapipe Hands，单图片模式
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # 单图片模式
    max_num_hands=1,             # 最多检测2只手
    model_complexity=1,
    min_detection_confidence=0.2
)

# 绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 读取图片
img_path = '/home/xuan/dianrobot/wjx/eye/get_r/imgs/test_camera_2.png'
depth_img_path = '/home/xuan/dianrobot/wjx/eye/get_r/imgs/test_camera_2_depth.png'
mask_path = '/home/xuan/dianrobot/wjx/eye/get_r/imgs/imgsdepth_mask.png'
image = cv2.imread(img_path)
depth_image = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print("图片读取失败")
    exit()

intrinsic = [[689.5, 0, 645.34], [0, 689.82, 355.85], [0, 0, 1]]
cam2world = [[-0.9901652346580831, 0.013433445840222654, 0.13925642034520688, -27.163789618795875], 
                                [0.006592239752229778, 0.9987537537350062, -0.049472030232080154, -593.2150728299232], 
                                [-0.13974745239020378, -0.048067472713806535, -0.9890198014283411, 834.0095420825568], 
                                [0.0, 0.0, 0.0, 1.0]]

def depth_pixel2cam_point3d(px, py, depth_image=None, intrinsic=None):
    """
    深度像素坐标(px, py)转换为相机坐标系下的三维坐标[x, y, z]
    参数:
        px, py: 像素坐标
        depth_image: 深度图（可选）
        intrinsic: 相机内参矩阵（可选，3x3）
    返回:
        [x_cam, y_cam, z_cam]: 相机坐标系下三维坐标
    """
    # 获取深度值
    if depth_image is not None:
        z_cam = depth_image[py, px]
    else:
        raise ValueError("必须提供depth_value")
    # 获取内参
    if intrinsic is not None:
        intrinsic = np.array(intrinsic)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
    else:
        raise ValueError("必须提供相机内参 intrinsic")
    # 计算相机坐标
    x_cam = (px - cx) * z_cam / fx
    y_cam = (py - cy) * z_cam / fy
    return [x_cam, y_cam, z_cam]



def transform_to_world_coordinates(cam2world, cam_point3d):
    """
    将相机坐标系下的三维点 cam_point3d 转换为世界坐标系下的三维点
    cam_point3d: [x, y, z]，相机坐标系下的点
    返回: [x_w, y_w, z_w]，世界坐标系下的点
    """
    # 获取相机到世界的旋转矩阵和平移向量
    # 假设 self.cam2world_R 为 3x3旋转矩阵，self.cam2world_T 为3x1平移向量
    cam2world = np.array(cam2world)
    cam2world_R = np.array(cam2world[:3, :3])
    cam2world_T = np.array(cam2world[:3, 3])    
    # 将相机坐标系下的点转换为齐次坐标
    cam_point3d_homogeneous = np.array(cam_point3d + [1.0])  # 添加齐次坐标
    # 进行坐标转换
    world_point_homogeneous = np.dot(cam2world_R, cam_point3d_homogeneous[:3]) + cam2world_T
    # 返回世界坐标系下的点
    return world_point_homogeneous.tolist()


def get_dot_pos_3d(results, dot_num, depth_img, intrinsic, cam2world, roi_size=5):
    '''
    获取手部关键点的世界坐标
    results: mediapipe hands 处理结果
    dot_num: 0-20号点
    depth_img: 深度图（numpy数组）
    camera: 相机对象，需有 depth_pixel2cam_point3d 方法
    intrinsic: 相机内参
    transform_to_world_coordinates: 坐标变换函数
    roi_size: 深度图ROI区域大小，默认5像素
    返回: (world_x, world_y, world_z)
    '''
    if results.multi_hand_landmarks is not None:
        pixel_landmark = results.multi_hand_landmarks[0].landmark
        # 获取关键点的像素坐标
        lm = pixel_landmark[dot_num]
        img_width, img_height = depth_img.shape[1], depth_img.shape[0]
        px = int(lm.x * img_width)
        py = int(lm.y * img_height)

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
            cam_point3d = SpaceMath.depth_pixel2cam_point3d(
                px, py, depth_value, intrinsic=intrinsic)
            cam_x, cam_y, cam_z = cam_point3d
            # 转换到世界坐标系
            world_point = transform_to_world_coordinates(cam2world, [cam_x, cam_y, cam_z])
            return world_point
        else:
            print("[WARN] ROI 区域无有效深度值")
            return None
    else:
        print("[INFO] 未检测到手部关键点")
        return None

# Mediapipe 处理
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
canvas = image.copy()
#canvas = mask_image.copy()  # 使用mask_image作为画布
# if len(mask_image.shape) == 2 or mask_image.shape[2] == 1:
#     # 单通道转三通道
#     canvas = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
# else:
#     canvas = mask_image.copy()
img_width, img_height = image.shape[1], image.shape[0]

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 绘制关键点和连接线
        mp_drawing.draw_landmarks(
            canvas,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        # 绘制关键点编号
        for idx, lm in enumerate(hand_landmarks.landmark):
            px = int(lm.x * img_width)
            py = int(lm.y * img_height)
            cv2.putText(canvas, str(idx), (px-10, py+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    # 打印所有点的世界坐标
    for i in range(21):
        world_pos = get_dot_pos_3d(results, i, depth_image, intrinsic, cam2world, roi_size=5)
        if world_pos is not None:
            print(f"点 {i} 的世界坐标: {world_pos}")
        else:
            print(f"点 {i} 的世界坐标: 无效")
    # 保存绘制结果
    cv2.imwrite('/home/xuan/dianrobot/wjx/eye/get_r/imgs/hand_landmarks.png', canvas)
    #cv2.imwrite('/home/xuan/dianrobot/wjx/eye/get_r/imgs/landmarks_onmaks.png', canvas)
    cv2.imshow('Hand Landmarks', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未检测到手掌关键点")

