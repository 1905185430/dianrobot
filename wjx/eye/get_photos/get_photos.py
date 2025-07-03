import cv2
import os
import time
from kyle_robot_toolbox.camera import Gemini335
import numpy as np

def get_unique_dir(base_dir):
    """
    如果base_dir已存在且不为空，则在同级创建新文件夹（base_dir_1, base_dir_2, ...）
    返回最终可用的文件夹路径
    """
    if not os.path.exists(base_dir) or len(os.listdir(base_dir)) == 0:
        return base_dir
    parent, name = os.path.split(base_dir.rstrip('/'))
    idx = 1
    while True:
        new_dir = os.path.join(parent, f"{name}{idx}")
        if not os.path.exists(new_dir) or len(os.listdir(new_dir)) == 0:
            return new_dir
        idx += 1

def auto_capture(root_dir, interval=2, max_count=10):
    """
    自动间隔采集深度图和RGB图，保存到指定文件夹
    :param root_dir: 根目录（会自动创建depth和rgbs子文件夹）
    :param interval: 采集间隔（秒）
    :param max_count: 最大采集次数
    """
    root_dir = get_unique_dir(root_dir)
    save_dir_depth = os.path.join(root_dir, "depth")
    save_dir_rgb = os.path.join(root_dir, "rgbs")
    os.makedirs(save_dir_depth, exist_ok=True)
    os.makedirs(save_dir_rgb, exist_ok=True)
    camera_1 = Gemini335(serial_num='CP1E54200015')
    camera_2 = Gemini335(serial_num='CP15641000AW')
    print(f"深度图保存目录: {save_dir_depth}")
    print(f"RGB图保存目录: {save_dir_rgb}")
    print("按下Enter键开始采集...")
    input()  # 等待用户按下Enter键开始采集
    print("开始自动采集，按Ctrl+C可中断")
    try:
        for i in range(max_count):
            img_bgr, depth_img = camera_1.read()
            t = time.strftime("%Y%m%d_%H%M%S")
            depth_path = os.path.join(save_dir_depth, f"depth{i:03d}_1.png")
            rgb_path = os.path.join(save_dir_rgb, f"rgb{i:03d}_1.png")
            depth_img = depth_img.astype(np.uint16)
            cv2.imwrite(depth_path, depth_img)
            cv2.imwrite(rgb_path, img_bgr)
            print(f"[{i+1}/{max_count}] 已保存: {depth_path}, {rgb_path}")

            img_bgr, depth_img = camera_2.read()
            depth_path = os.path.join(save_dir_depth, f"depth{i:03d}_2.png")
            rgb_path = os.path.join(save_dir_rgb, f"rgb{i:03d}_2.png")
            depth_img = depth_img.astype(np.uint16)
            cv2.imwrite(depth_path, depth_img)
            cv2.imwrite(rgb_path, img_bgr)
            print(f"[{i+1}/{max_count}] 已保存: {depth_path}, {rgb_path}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("采集中断")
    finally:
        camera_1.release()
        camera_2.release()
        print("相机已释放")

if __name__ == "__main__":
    auto_capture(
        root_dir="/home/xuan/dianrobot/wjx/eye/get_photos/data",
        interval=0.3,
        max_count=100
    )