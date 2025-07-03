from SingleImageProsessor import SingleImageProcessor
import cv2
import os
import numpy as np
import csv
if __name__ == "__main__":
    # 载入/home/xuan/dianrobot/wjx/eye/get_photos/data1 目录下的图片
    # 深度图在depth子目录下，RGB图在rgbs子目录下
    # 以_1结尾的为相机1的图片，以_2结尾的为相机2的图片
    # 例如：depth/depth000_1.png, rgbs/rgb000_1.png
    #      depth/depth000_2.png, rgbs/rgb000_2.png
    # 读取这些图片并处理
    # 检测文件夹下图片数量
    root_dir = "/home/xuan/dianrobot/wjx/eye/get_photos/data2"
    depth_dir = os.path.join(root_dir, "depth")
    rgb_dir = os.path.join(root_dir, "rgbs")
    if not os.path.exists(depth_dir) or not os.path.exists(rgb_dir):
        print(f"深度图或RGB图目录不存在: {depth_dir} 或 {rgb_dir}")
        exit(1)
    depth_images_1 = sorted([f for f in os.listdir(depth_dir) if f.endswith('_1.png')])
    rgb_images_1 = sorted([f for f in os.listdir(rgb_dir) if f.endswith('_1.png')])
    depth_images_2 = sorted([f for f in os.listdir(depth_dir) if f.endswith('_2.png')])
    rgb_images_2 = sorted([f for f in os.listdir(rgb_dir) if f.endswith('_2.png')])
    cam2world_1 = [[-0.998625444888728, 0.036733716694060184, -0.037387897488769944, -180.37320672771654], 
                   [0.039552924083290045, 0.9961925961217319, -0.07769091085001648, -580.9520723024289], 
                   [0.03439167075400293, -0.07906292108241536, -0.9962762004046188, 567.4327229504052], 
                   [0.0, 0.0, 0.0, 1.0]]
    cam2world_2 = [[0.9992354675776733, 0.017822376363469367, 0.034797172810269915, 268.43711333982395], 
                   [0.015513072578121312, -0.9977291861574156, 0.0655424722515648, -535.6764685371061], 
                   [0.03588627751682919, -0.06495255183916988, -0.9972428696639375, 550.1928515979271], 
                   [0.0, 0.0, 0.0, 1.0]]
    bias_2 = [0, 0, 0, 1, 1, 23, 30, 30, 30,30 , 30, 30, 30, 30, 30, 30, 17, 8, 3]
    # 打开CSV文件准备写入
    csv_path = os.path.join(root_dir, "gripper_positions.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写表头 X,Y,Z,RX,RY,RZ
        csv_writer.writerow(['X', 'Y', 'Z', 'RX', 'RY', 'RZ'])
        for i in range(len(depth_images_1)):

            # 创建处理器实例
            processor_1 = SingleImageProcessor(cam2world=cam2world_1)
            processor_2 = SingleImageProcessor(cam2world=cam2world_2, bias= bias_2[i])
            # 处理图片
            gripper_pos1 = processor_1.run(rgb_image_path=os.path.join(rgb_dir, rgb_images_1[i]),
                                           depth_image_path=os.path.join(depth_dir, depth_images_1[i]))
            gripper_pos2 = processor_2.run(rgb_image_path=os.path.join(rgb_dir, rgb_images_2[i]),
                                           depth_image_path=os.path.join(depth_dir, depth_images_2[i]))

            # 写入CSV
            csv_writer.writerow([
                *(gripper_pos2 if gripper_pos2 is not None else [None, None, None, None, None, None]),
            ])
    print(f"结果已保存到 {csv_path}")