'''
YoloV8目标检测
(瓶盖检测)
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import os
import time
import numpy as np
import cv2
# 阿凯机器人工具箱
from kyle_robot_toolbox.camera import Gemini335
from kyle_robot_toolbox.yolov8 import YoloV8Detect

# 创建摄像头
camera = Gemini335()

# 创建窗口
cv2.namedWindow('canvas', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# 模型初始化
print("[INFO] 开始YoloV8模型加载")
# 模型路径
model_path = os.path.join(os.path.abspath("."), "weights", "bottle_cap_yolov8n.pt")
print(f"模型路径: {model_path}")
# 载入目标检测模型(使用绝对路径)
model = YoloV8Detect(model_path)
# 配置模型参数(可选)
# - 图像尺寸(必须是32的倍数)
model.IMAGE_SIZE= 1088
# - 置信度
model.CONFIDENCE = 0.5
# - IOU 
model.IOU = 0.6
print("[INFO] 完成YoloV8模型加载")

fps = 0
while True:
    t_start = time.time() # 开始计时
    # 清空缓冲区
    img = camera.read_color_img()
    # 获取工作台的图像
    if img is None:
        print("[Error] USB摄像头获取失败")
        break
    
    # YoloV5 目标检测
    canvas, class_id_list, xyxy_list, conf_list = model.detect(img)
    
    # 结束计时
    t_end = time.time() 
    # 添加fps显示
    ratio = 0.1
    fps = int((1-ratio)*fps + ratio*(1.0/(t_end - t_start)))
    cv2.putText(canvas, text="FPS: {}".format(fps), org=(50, 50), \
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1,\
        lineType=cv2.LINE_AA, color=(0, 0, 255))
    # 可视化
    cv2.imshow("canvas", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()