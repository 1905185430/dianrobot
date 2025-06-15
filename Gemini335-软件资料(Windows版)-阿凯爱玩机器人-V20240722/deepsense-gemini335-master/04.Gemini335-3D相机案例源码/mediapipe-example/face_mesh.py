'''
人脸特征点检测-MediaPipe
----------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
购买链接: https://item.taobao.com/item.htm?id=677075846402
'''
import numpy as np
import cv2
import mediapipe as mp
from kyle_robot_toolbox.camera import Gemini335

# 人脸Mesh检测模型
mp_face_mesh = mp.solutions.face_mesh

# 绘制工具
mp_drawing = mp.solutions.drawing_utils
# 绘制样式
mp_drawing_styles = mp.solutions.drawing_styles
# 关键点绘制样式 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 创建相机对象
camera = Gemini335()
face_mesh =  mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cv2.namedWindow('MediaPipe Face Mesh', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

while camera.capture.isOpened():
    img_bgr = camera.read_color_img()
    if img_bgr is None:
        continue
    # 为了提高性能， 将图像标记为只读模式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = face_mesh.process(img_rgb)
    # 检测结果可视化
    canvas = np.copy(img_bgr)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # 绘制人脸Mesh网格
        mp_drawing.draw_landmarks(
            image=canvas,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        # 绘制人脸边缘
        mp_drawing.draw_landmarks(
            image=canvas,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        # 绘制虹膜(眼睛+眉毛)
        mp_drawing.draw_landmarks(
            image=canvas,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # 为了有照镜子的感觉， 水平镜像一下图像
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(canvas, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
camera.release()
