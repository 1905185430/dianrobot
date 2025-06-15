'''
人脸关键点检测-MediaPipe
'''
import cv2
import mediapipe as mp

# 人脸Mesh检测模型
mp_face_mesh = mp.solutions.face_mesh

# 绘制工具
mp_drawing = mp.solutions.drawing_utils
# 绘制样式
mp_drawing_styles = mp.solutions.drawing_styles
# 关键点绘制样式 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
face_mesh =  mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

while cap.isOpened():
    ret, img_bgr = cap.read()
    if not ret:
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
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
