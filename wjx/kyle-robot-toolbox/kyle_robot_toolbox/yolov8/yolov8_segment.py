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

class YoloV8Segment:
	'''YoloV8实例分割模型'''
	# 模型参数
	# - 图像尺寸
	IMAGE_SIZE = 1088
	# - 置信度
	CONFIDENCE = 0.6
	# - IOU
	IOU = 0.6
	
	def __init__(self, model_path, image_size=None, confidence=None, iou=None):
		'''初始化'''
		# 设置图像尺寸
		if image_size is not None:
			self.IMAGE_SIZE = image_size
		# 置信度
		if confidence is not None:
			self.CONFIDENCE = confidence
		# 设置IOU
		if iou is not None:
			self.IOU = iou
		
		# 设置模型路径
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
	
	def segment(self, img, canvas=None, \
			draw_mask=True, draw_box=False, draw_label=False):
		'''实例分割'''
		if canvas is None:
			canvas = np.copy(img)
		# 注意事项:
		# - imgsz: 输入图像尺寸, 必须为32的倍数
		# - conf: 置信度
		# - iou: IOU阈值
		results = self.model(img, imgsz=self.IMAGE_SIZE, \
			conf=self.CONFIDENCE, iou=self.IOU)
		result = results[0]
		# 提取BOX信息
		boxes_list = np.float64(result.boxes.data.tolist())
		canvas = np.copy(img)
		if len(boxes_list) == 0:
			return canvas, [], [], [], []
		else:
			xyxy_list = boxes_list[:, :4].astype(np.int32).tolist()
			conf_list = boxes_list[:, 4]
			class_id_list = boxes_list[:, 5].astype(np.int32).tolist()
		# 提取Mask的信息
		# - 获取掩码列表
		obj_mask_list = np.uint8(result.masks.data.tolist()) * 255
		# 将Mask转换为彩图的尺寸
		img_height, img_width = img.shape[:2]
		obj_mask_list_resized = []
		
		if len(result.masks) != 0:
			# 案例
			# 彩图分辨率 1280 * 720, 等比例缩放后是 1088 * 612
			# 传入模型的时候，会将尺寸填补到32的倍数 1088*640
			# 实际得到的Mask也是1088*640
			# 因此将Mask还原到原尺寸的时候，需要将Mask顶部还有底部去掉一部分再去等比例缩放
			# 获取原来输入的图像尺寸
			orig_height, orig_width =  result.masks[0].orig_shape
			# 获取YoloV8示例分割模型输出的图像尺寸
			out_height, out_width = list(result.masks[0].shape)[1:]
			# 计算理论上height等比例缩放的尺寸
			ratio = out_width / orig_width
			# 计算如果是等比例缩放的高度
			real_out_height = int(orig_height * ratio)
			# 计算高度偏移量
			offset_h = int((out_height - real_out_height)/2)

			# print(f"offset_h = {offset_h}")
			
			for mask in obj_mask_list:
				if offset_h == 0:
					# 计算对齐缩放后的mask
					mask_resize = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
				else:
					# 计算对齐缩放后的mask
					mask_resize = cv2.resize(mask[offset_h:-offset_h,:], (img_width, img_height), interpolation=cv2.INTER_NEAREST)
				obj_mask_list_resized.append(mask_resize)
		# 可视化-绘制Mask
		if draw_mask:
			# 创建Mask掩码画布
			mask_canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
			for class_id, mask in zip(class_id_list, obj_mask_list_resized):
				# mask_canvas[mask != 0] = [np.random.randint(255) for i in range(3)]
				mask_canvas[mask != 0] = self.colors[class_id]

			# 彩图与Mask画布叠加
			canvas = cv2.addWeighted(canvas, 1.0, mask_canvas, 0.5, 0.2)

		# 可视化-绘制BOX
		if draw_box or draw_label:
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
		
		return canvas, class_id_list, obj_mask_list_resized, xyxy_list, conf_list
