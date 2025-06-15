'''
YoloV8目标检测模型
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
# YoloV8
from ultralytics import YOLO

class YoloV8Detect:
	'''YoloV8目标检测模型'''
	# 模型参数
	# - 图像尺寸
	IMAGE_SIZE = 1088
	# - 置信度
	CONFIDENCE = 0.5
	# - IOU
	IOU = 0.6
	
	def __init__(self, model_path):
		'''初始化'''
		self.model_path = model_path
		self.model = YOLO(model_path)
		# 解析模型个数
		self.class_num = len(self.model.names.keys())
		# 模型名称字典
		self.class_name_dict = self.model.names
		# 随机生成每个类别的颜色
		self.colors = [[np.random.randint(0, 255) for _ in range(3)]  for class_id in range(self.class_num)]
	
	def detect(self, img, canvas=None, \
			draw_box=True, draw_label=True):
		'''目标检测'''
		if canvas is None:
			canvas = np.copy(img)
		# 注意事项:
		# - imgsz: 输入图像尺寸, 必须为32的倍数
		# - conf: 置信度
		# - iou: IOU阈值
		results = self.model(img, imgsz=self.IMAGE_SIZE, \
			conf=self.CONFIDENCE, iou=self.IOU)
		# 转换为list
		boxes_list = np.float64(results[0].boxes.data.tolist())
		canvas = np.copy(img)
		if len(boxes_list) == 0:
			return canvas, [], [], []
		else:
			xyxy_list = boxes_list[:, :4].astype(np.int32).tolist()
			conf_list = boxes_list[:, 4]
			class_id_list = boxes_list[:, 5].astype(np.int32).tolist()
		
		
		for box in boxes_list:
			x1, y1, x2, y2, conf, class_id = box
			pt1 = (int(x1), int(y1))
			pt2 = (int(x2), int(y2))
			class_id = int(class_id)
			label = self.class_name_dict[class_id]
			self.plot_one_box([x1,y1, x2, y2], canvas, label=label, \
				color=self.colors[class_id],\
				 	line_thickness=3, \
					draw_box=draw_box, draw_label=draw_label)

		return canvas, class_id_list, xyxy_list, conf_list
		
	def plot_one_box(self, x, img, color=None, label=None, line_thickness=None, \
			draw_box=True, draw_label=True):
		''''绘制矩形框+标签'''
		tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
		color = color or [random.randint(0, 255) for _ in range(3)]
		c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
		if draw_box:
			cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
		if draw_label:
			tf = max(tl - 1, 1)  # font thickness
			t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
			c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
			cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
			cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
