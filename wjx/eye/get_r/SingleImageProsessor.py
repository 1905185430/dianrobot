'''
该文件包含了处理单张图像的相关代码
主要用于读取图片、处理图片以及获取图片中的特征点等功能
从手的姿态图片转化到机械臂夹爪的姿态
'''

import cv2
import mediapipe as mp
import numpy as np
import SpaceMath as Spmath
import depth_img_process as DP
import pickle

class SingleImageProcessor:
    '''单张图像处理器'''
    
    def __init__(self, image=None, depth_image=None, cam2world=None, bias = 0):
        '''初始化处理器'''
        # 图片相关
        self.rgb_image = image
        self.depth_image = depth_image
        # 深度图和h w
        self.depth_height = None
        self.depth_width = None
        # rgb图和h w
        self.rgb_height = None
        self.rgb_width = None
        # 手部深度图mask
        self.depth_mask = None
        # 加入mask后的深度图
        self.depth_masked_image = None
        # hand_points一共21个点
        self.hand_points = [None] * 21
        self.results = None
        # 深度图和彩色图之间的偏移
        self.bias = bias  # 偏移量，默认为0
        # 手部关键点位于相机图片内的像素坐标
        self.hand_points_pixel = [None] * 21  # 用于存储手部关键点的像素坐标
        # 转化到深度图的坐标
        self.hand_points_depth2d = [None] * 21  # 用于存储手部关键点的深度坐标

        # 初始化 Mediapipe Hands，单图片模式
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,      # 单图片模式
            max_num_hands=1,             # 最多检测1只手
            model_complexity=1,
            min_detection_confidence=0.2
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles


        # 相机的参数
        self.intrinsic = [[689.5, 0, 645.34], [0, 689.82, 355.85], [0, 0, 1]]
        # self.cam2world = [[-0.9901652346580831, 0.013433445840222654, 0.13925642034520688, -27.163789618795875], 
        #                   [0.006592239752229778, 0.9987537537350062, -0.049472030232080154, -593.2150728299232], 
        #                   [-0.13974745239020378, -0.048067472713806535, -0.9890198014283411, 834.0095420825568], 
        #                   [0.0, 0.0, 0.0, 1.0]]
        # cam2
        if cam2world is None:
            # 默认相机到世界的变换矩阵
            self.cam2world = [[0.9992354675776733, 0.017822376363469367, 0.034797172810269915, 268.43711333982395], 
                              [0.015513072578121312, -0.9977291861574156, 0.0655424722515648, -535.6764685371061], 
                              [0.03588627751682919, -0.06495255183916988, -0.9972428696639375, 550.1928515979271], 
                              [0.0, 0.0, 0.0, 1.0]]
        else:
            self.cam2world = cam2world
        # 拟合3D点集的平面ax+by+cz+d=0
        # 点0到8的拟合平面
        self.plane = [None] * 4  # 平面方程参数 (a, b, c, d)

        # 点4指向点8的向量，作为工件坐标系的x轴
        self.vector_4_to_8 = [None] * 3  


        # 工件坐标系的原点
        self.origin = []
        # 工件坐标系的x轴
        self.x_axis = []
        # 工件坐标系的y轴
        self.y_axis = []
        # 工件坐标系的z轴
        self.z_axis = []   
        # 工件坐标系到世界坐标系的旋转矩阵
        self.tool2world_R = None
        # 工件坐标系到世界坐标系的平移向量
        self.tool2world_T = None
        # 工件坐标系到世界坐标系的变换矩阵
        self.tool2world_TF = None
        
        # rx, ry, rz 欧拉角
        self.rx = None
        self.ry = None  
        self.rz = None

        # 夹爪pos
        self.gripper_pos = []

        # 绘图相关
        



    # 加载图像和深度图像
    def load_rgb_image(self, image_path):
        '''加载图像'''
        self.rgb_image = cv2.imread(image_path)
        if self.rgb_image is not None:
            self.rgb_height, self.rgb_width = self.rgb_image.shape[:2]
            print(f"加载图像成功: {image_path}, 尺寸: {self.rgb_width}x{self.rgb_height}")
        else:
            print(f"加载图像失败: {image_path}")
            self.rgb_image = None

    def load_depth_image(self, depth_image_path):
        '''加载深度图像'''
        self.depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if self.depth_image is not None:
            self.depth_height, self.depth_width = self.depth_image.shape[:2]
            print(f"加载深度图像成功: {depth_image_path}, 尺寸: {self.depth_width}x{self.depth_height}")
        else:
            print(f"加载深度图像失败: {depth_image_path}")
            self.depth_image = None
        
    def load_images(self, image_path, depth_image_path):
        '''加载RGB和深度图像'''
        self.load_rgb_image(image_path)
        self.load_depth_image(depth_image_path)

    def save_hand_points_pixel(self, save_path):
        '''保存手部关键点的像素坐标到文件'''
        if self.hand_points_pixel is None:
            raise ValueError("请先获取手部关键点的像素坐标")
        with open(save_path, 'wb') as f:
            pickle.dump(self.hand_points_pixel, f)
        print(f"手部关键点像素坐标已保存到 {save_path}")

    def rgb2depth_pixel(self, pixel):
        '''将RGB图像的像素坐标转换为深度图像的像素坐标'''
        if self.rgb_image is None or self.depth_image is None:
            raise ValueError("请先加载RGB和深度图像")
        if pixel[0] < 0 or pixel[0] >= self.rgb_width or pixel[1] < 0 or pixel[1] >= self.rgb_height:
            raise ValueError("像素坐标超出RGB图像范围")
        x_rgb, y_rgb = pixel
        # 计算深度图像的像素坐标
        depth_w, depth_h = self.depth_width, self.depth_height
        rgb_w, rgb_h = self.rgb_width, self.rgb_height
        x_depth = int(x_rgb * depth_w / rgb_w)
        y_depth = int(y_rgb * depth_h / rgb_h)
        return (x_depth, y_depth)
    


    # mask相关
    def draw_depth_mask(self):
        '''
        根据手部关键点生成深度图的mask
        使用点0作为种子点进行区域生长
        '''
        points = np.array([self.hand_points_pixel[i] for i in [0,1,2,3,5,6]])
        points = points + np.array([self.bias, 0]) # 偏移像素
        mask = DP.region_grow_by_depth(points, self.depth_image, max_diff=50)
        cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/imgsdepth_mask.png", mask)
        self.depth_mask = mask
        self.depth_masked_image = cv2.bitwise_and(self.depth_image, self.depth_image, mask=mask)

    # 绘图函数
    def draw_workpiece_axes_on_image(self, axis_length=100, show=False, window_name="Workpiece Axes"):
        """
        在图像上绘制工件坐标系
        参数:
            img: 输入RGB图像
            T_workpiece2world: 工件到世界的4x4变换矩阵
            T_cam2world: 相机到世界的4x4变换矩阵
            K: 相机内参矩阵 (3x3)
            axis_length: 坐标轴长度（工件坐标系下单位）
            show: 是否直接显示图像
            window_name: 显示窗口名
        返回:
            绘制了工件坐标系的图像
        """
        if self.rgb_image is None:
            raise ValueError("请先加载RGB图像")
        
        img = self.rgb_image.copy()
        if not hasattr(self, 'tool2world_TF') or self.tool2world_TF is None:
            raise ValueError("请先计算工具坐标系到世界坐标系的变换矩阵")
        T_workpiece2world = self.tool2world_TF
        T_cam2world = np.array(self.cam2world)
        K = np.array(self.intrinsic)
        
        # 工件坐标系原点和三轴（齐次坐标）
        origin_w = np.array([0, 0, 0, 1])
        x_axis_w = np.array([axis_length, 0, 0, 1])
        y_axis_w = np.array([0, axis_length, 0, 1])
        z_axis_w = np.array([0, 0, axis_length, 1])

        # 世界到相机的逆变换
        T_world2cam = np.linalg.inv(T_cam2world)

        # 工件坐标系下的点变换到相机坐标系
        origin_c = T_world2cam @ (T_workpiece2world @ origin_w)
        x_axis_c = T_world2cam @ (T_workpiece2world @ x_axis_w)
        y_axis_c = T_world2cam @ (T_workpiece2world @ y_axis_w)
        z_axis_c = T_world2cam @ (T_workpiece2world @ z_axis_w)

        def project_point(pt3d, K):
            X, Y, Z = pt3d[:3]
            u = int(K[0, 0] * X / Z + K[0, 2])
            v = int(K[1, 1] * Y / Z + K[1, 2])
            return (u, v)

        origin_uv = project_point(origin_c, K)
        x_uv = project_point(x_axis_c, K)
        y_uv = project_point(y_axis_c, K)
        z_uv = project_point(z_axis_c, K)

        img = cv2.line(img, origin_uv, x_uv, (0,0,255), 4)  # X轴红
        img = cv2.line(img, origin_uv, y_uv, (0,255,0), 4)  # Y轴绿
        img = cv2.line(img, origin_uv, z_uv, (255,0,0), 4)  # Z轴蓝
        cv2.putText(img, 'X', x_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(img, 'Y', y_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(img, 'Z', z_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(img, 'Origin', origin_uv, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/workpiece_axes.png", img)  # 保存图像

        if show:
            cv2.imshow(window_name, img)
        # 返回绘制了工件坐标系的图像
        return img



    # 生成检测结果results
    def images_to_results(self):
        '''将图像转换为mediapipe处理结果'''
        bgr_img = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(bgr_img)
        if self.results == None:
            print("[WARN] 未检测到手部关键点")
        return self.results
    
    def get_hand_points_pixel(self):
        '''获取手部关键点的像素坐标'''
        if self.results is None:
            raise ValueError("请先调用 images_to_results() 方法获取处理结果")
        if self.results.multi_hand_landmarks is not None:
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            img_width, img_height = self.rgb_image.shape[1], self.rgb_image.shape[0]
            for i, lm in enumerate(pixel_landmark):
                px = int(lm.x * img_width)
                py = int(lm.y * img_height)
                self.hand_points_pixel[i] = [px, py]
            return self.hand_points_pixel
        else:
            print("[INFO] 未检测到手部关键点")
            return None

    def get_hand_point(self, index, roi_size=5):
        '''获取指定手部关键点，支持在像素范围内取深度均值'''
        if self.results is None:
            raise ValueError("请先调用 images_to_results() 方法获取处理结果")
        if index < 0 or index >= len(self.hand_points):
            raise ValueError(f"索引 {index} 超出范围，必须在 0 到 {len(self.hand_points) - 1} 之间")
        if self.results.multi_hand_landmarks is not None:
            
            pixel_landmark = self.results.multi_hand_landmarks[0].landmark
            lm = pixel_landmark[index]
            img_width, img_height = self.depth_image.shape[1], self.depth_image.shape[0]
            px = int(lm.x * img_width)
            py = int(lm.y * img_height)
            px = px + self.bias  # 偏移像素
            if self.depth_image is not None:
                #roi = self.depth_image[y1:y2, x1:x2]
                (px, py) ,roi = DP.find_valid_square((px, py), self.depth_masked_image, img_height, img_width, size=roi_size, step=3, min_ratio=0.8, max_iter=20)
                #print(roi)
                valid = roi[roi > 0]
                if valid.size == 0:
                    print("[WARN] ROI 区域无有效深度值")
                    return None
                depth_mean = np.mean(valid)
                #print(f"点 {index} 的深度均值: {depth_mean}")
                cam_point3d = Spmath.depth_pixel2cam_point3d(px, py, depth_mean, self.intrinsic)
                world_point = Spmath.Cam2World(self.cam2world, cam_point3d)
                self.hand_points[index] = world_point
                print(f"点 {index} 的世界坐标: {world_point}")
                self.hand_points_pixel[index] = [px, py]  # 保存像素坐标
                return world_point
            else:
                print("[WARN] ROI 区域无有效深度值")
                return None
        else:
            print("[INFO] 未检测到手部关键点")
            return None
        
    def update_all_hand_points(self):
        '''更新所有手部关键点'''
        if self.results is None:
            raise ValueError("请先调用 images_to_results() 方法获取处理结果")
        if self.results.multi_hand_landmarks is None:
            print("[INFO] 未检测到手部关键点")
            return
        # 获取第一个手的关键点
        for i in range(len(self.hand_points)):  
            self.get_hand_point(i, 30)
            


    def get_4_to_8_vector(self):
        '''获取点4到点8的向量在plane上的投影'''
        if self.hand_points[3] is None or self.hand_points[6] is None:
            print("[WARN] 点4或点8的世界坐标未获取")
            return None
        p3 = np.array(self.hand_points[3])
        p6 = np.array(self.hand_points[6])
        print(f"点4的世界坐标: {p3}")
        print(f"点8的世界坐标: {p6}")
        vector = p3 - p6
        print(f"点4到点8的向量: {vector}")
        vector = vector / np.linalg.norm(vector)  # 单位化向量
        print(f"单位化后的向量: {vector}")
        vector = Spmath.project_vector_on_plane(self.plane, vector) # 投影到平面上
        self.vector_4_to_8 = vector
        self.x_axis = vector  # 将x轴设置为点4到点8的向量
        return vector
    
    def get_from_0_to_8_plane(self):
        '''获取从点0到点8的平面方程'''
        # 选取0到9的非None作为拟合
        points = np.array([self.hand_points[i] for i in [1,2,3,5,6,7] if self.hand_points[i] is not None])
        a, b, c, d = Spmath.fit_plane(points)
        self.plane = [a, b, c, d]
        print(f"平面方程: {a}x + {b}y + {c}z + {d} = 0")
        return self.plane
    
    def get_z_axis_vector(self):
        '''获取平面内与x轴垂直且与y轴夹角小于90度的向量'''
        if self.plane is None:
            print("[WARN] 平面方程未计算")
            return None
        a = self.x_axis
        dot_0 = self.hand_points[0]
        dot_4 = self.hand_points[5]
        if dot_0 is None or dot_4 is None:
            print("[WARN] 点0或点4的世界坐标未获取")
            return None
        b = np.array(dot_4) - np.array(dot_0)  # 点4到点0的向量
        b = b / np.linalg.norm(b)  
        b = Spmath.project_vector_on_plane(self.plane, b) 
        perp = Spmath.perp_vector_on_plane_with_ref(self.plane, a, b)
        self.z_axis = perp
        return perp
    
    def get_y_axis_vector(self):
        '''
        获取y轴向量
        '''
        if self.plane is None:
            print("[WARN] 平面方程未计算")
            return None
        if not self.x_axis or not self.z_axis:
            print("[WARN] x轴或z轴向量未计算")
            return None
        # z轴向量为x轴和y轴的叉乘
        x_axis = np.array(self.x_axis)
        z_axis = np.array(self.z_axis)  
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)  # 单位化向量
        self.y_axis = y_axis
        return y_axis
    
    def update_all_axes(self):
        '''更新所有轴向量'''
        self.get_from_0_to_8_plane()
        self.get_4_to_8_vector()
        self.get_z_axis_vector()
        self.get_y_axis_vector()
        
        print("所有轴向量已更新")
        print(f"x轴: {self.x_axis}")
        print(f"y轴: {self.y_axis}")
        print(f"z轴: {self.z_axis}")

    def get_tool2world_transformation(self):
        '''
        获取工具坐标系到世界坐标系的变换矩阵
        返回旋转矩阵和位移向量
        '''
        if self.x_axis is None or self.y_axis is None or self.z_axis is None:
            print("轴向量未设置")
            return None
        
        # 工具坐标系到世界坐标系的旋转矩阵
        R = np.array([self.x_axis, self.y_axis, self.z_axis]).T
        
        # 工具坐标系原点在世界坐标系中的位置
        dot1 = np.array(self.hand_points[3])
        dot2 = np.array(self.hand_points[6])
        tool_origin = (dot1 + dot2) / 2  # 工具坐标系原点取点4和点8的中点
        self.origin = tool_origin.tolist()  # 保存原点位置

        # 工具坐标系到世界坐标系的平移向量
        T = tool_origin
        # 工具坐标系到世界坐标系的变换矩阵
        self.tool2world_TF = np.eye(4)
        self.tool2world_TF[:3, :3] = R
        self.tool2world_TF[:3, 3] = T
        self.tool2world_R = R
        self.tool2world_T = T
        print(f"工具坐标系到世界坐标系的变换矩阵:\n{self.tool2world_TF}")
        return self.tool2world_TF

    def get_rxryrz_from_rotation_matrix(self):
        '''
        从旋转矩阵获取欧拉角（rx, ry, rz）
        返回: [rx, ry, rz]
        '''
        if self.tool2world_R is None:
            print("[WARN] 工具坐标系到世界坐标系的旋转矩阵未计算")
            return None
        self.rx, self.ry, self.rz = Spmath.get_rxryrz_from_R(self.tool2world_R)
        print(f"欧拉角: rx={self.rx}, ry={self.ry}, rz={self.rz}")
        return self.rx, self.ry, self.rz


    def update_gripper_position(self):
        '''
        更新夹爪位置
        gripper_pos: 夹爪在世界坐标系中的位置 [x, y, z, rx, ry, rz]
        '''
        self.gripper_pos = self.tool2world_T
        self.gripper_pos = self.gripper_pos.tolist()
        self.gripper_pos.append(self.rx)
        self.gripper_pos.append(self.ry)
        self.gripper_pos.append(self.rz)
        print(f"夹爪位置已更新: {self.gripper_pos}")

    def draw_and_show_hand_landmarks(self):
        '''
        绘制手部关键点、连接线、编号，并打印每个点的世界坐标
        '''
        if self.rgb_image is None or self.depth_image is None:
            print("[WARN] 图像未加载")
            return
        if self.results is None:
            print("[WARN] 未检测到手部关键点")
            return

        canvas = self.rgb_image.copy()
        img_width, img_height = self.rgb_image.shape[1], self.rgb_image.shape[0]
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # 绘制关键点和连接线
                self.mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                # 绘制关键点编号
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px = int(lm.x * img_width)
                    py = int(lm.y * img_height)
                    cv2.putText(canvas, str(idx), (px-10, py+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Hand Landmarks', canvas)
            cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/hand_landmarks.png", canvas)
            # cv2.waitKey(0)
            # cv2.destroyWindow('Hand Landmarks')
        else:
            print("[INFO] 未检测到手部关键点")


    def draw_depth_mask_on_image(self):
        '''
        在masked_depth_image上绘制手部关键点的像素坐标（色彩可视化）
        只绘制点0到点8的关键点
        '''
        points = [self.hand_points_pixel[i] for i in range(9) if self.hand_points_pixel[i] is not None]
        if self.depth_masked_image is None:
            print("[WARN] 深度图mask未生成")
            return
        # 归一化并伪彩色
        depth = self.depth_masked_image
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        canvas = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        for i, (px, py) in enumerate(points):
            px = px + self.bias  # 偏移像素
            if px is not None and py is not None:
                px, py = self.rgb2depth_pixel((px, py))  # 转换为深度图像的像素坐标
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(canvas, str(i), (px + 10, py + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite("/home/xuan/dianrobot/wjx/eye/get_r/imgs/depth_mask_with_hand_points.png", canvas)
        # cv2.imshow("depth_mask_with_hand_points", canvas)
        # cv2.waitKey(0)

    def run(self, rgb_image_path=None, depth_image_path=None):
        # rgb_image_path = "/home/xuan/dianrobot/wjx/eye/get_photos/data2/rgbs/rgb000_2.png"
        # depth_image_path = "/home/xuan/dianrobot/wjx/eye/get_photos/data2/depth/depth000_2.png"
        print("加载图像和深度图像...")
        self.load_images(rgb_image_path, depth_image_path)
        self.images_to_results()
        if self.get_hand_points_pixel() == None:
            print("未检测到手部关键点")
            return None
        self.draw_depth_mask()
        self.draw_depth_mask_on_image()
        self.save_hand_points_pixel("/home/xuan/dianrobot/wjx/eye/get_r/imgs/hand_points_pixel.pkl")
        self.draw_and_show_hand_landmarks()
        self.update_all_hand_points()
        self.update_all_axes()
        self.get_tool2world_transformation()
        self.draw_workpiece_axes_on_image(axis_length=50, show=True, window_name="Workpiece Axes")
        self.get_rxryrz_from_rotation_matrix()
        self.update_gripper_position()
        return self.gripper_pos


if __name__ == "__main__":
    # 示例用法
    processor = SingleImageProcessor(bias = 0)
    processor.run()