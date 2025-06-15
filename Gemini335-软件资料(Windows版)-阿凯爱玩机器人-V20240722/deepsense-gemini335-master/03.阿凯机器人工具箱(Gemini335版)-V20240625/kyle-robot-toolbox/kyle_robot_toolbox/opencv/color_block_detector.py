'''
多元高斯分布 色块识别
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np
import cv2

def is_legal_rect(rect, roi_min_size=20, roi_max_size=200):
    '''判断矩形框是否合法'''
    x, y, w, h = rect
    # 过滤掉小噪点
    if w < roi_min_size or h < roi_min_size or w > roi_max_size or h > roi_max_size:
        return False
    return True

def detect_color_block(img_bgr, hsv_lowerb, hsv_upperb, \
                    roi_min_size=20, roi_max_size=200):
    '''检测色块'''
    # 创建画布
    canvas = np.copy(img_bgr)
    # 将图像转换为HSV空间
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 根据HSV阈值获取Mask
    mask = cv2.inRange(img_hsv, np.uint8(hsv_lowerb), np.uint8(hsv_upperb))
    # 数学形态学操作
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.dilate(mask, np.ones((7, 7)))
    # 连通域检测
    *_, contours, hierarchy = cv2.findContours(image=mask, \
                mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # 外接矩形列表
    rect_list = []
    # 计算外接矩形
    for cnt in contours:
        # 计算外界矩形
        rect = cv2.boundingRect(cnt)
        # 判断是否合法
        if is_legal_rect(rect, roi_min_size, roi_max_size):
            # 添加外接矩形到列表
            rect_list.append(rect)
            # 绘制矩形框
            x, y, w, h = rect
            color = [0, 0, 255]
            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, thickness=4)
    return rect_list, mask, canvas

# import time
# import glob
# import pickle
# import cv2
# import numpy as np
# from scipy.stats import multivariate_normal

# class ColorBlockDetector:
# 	'''色块检测器'''
# 	# 颜色的名称
# 	color_names = ['red', 'green', 'blue', 'yellow'] 
# 	# 可视化-颜色的BGR值
# 	color_bgr = {
# 		'red': (0, 0, 255),
# 		'green': (0, 255, 0),
# 		'blue': (255, 0, 0),
# 		'yellow': (0, 255, 255)
# 	}
# 	# ROI区域的最小尺寸
# 	roi_min_size = 50
# 	# 图像二值化的概率密度阈值
# 	bin_pdf_threshold = 0.0000001 
# 	def __init__(self, data_folder="./data", config_folder='./config', is_update=True):
# 		self.data_folder = data_folder
# 		self.config_folder = config_folder
# 		if is_update:
# 			# 每次载入之前都更新一次
# 			self.color_rgb_statis()
# 		# 载入图像信息
# 		self.load_color_info()
		
# 	def color_rgb_statis(self):
# 		'''BGR颜色空间下的颜色统计'''
# 		conv_bgr_dict = {}
# 		mean_bgr_dict = {}
		
# 		# 遍历所有的颜色
# 		for color_name in ['red', 'green', 'blue', 'yellow']:
# 			img_paths = glob.glob(f'{self.data_folder}/color_block_roi/{color_name}/*.png')
# 			bgr_list = []
# 			for img_path in img_paths:
# 				img = cv2.imread(img_path)
# 				bgr_list += list(img.reshape(-1, 3))
# 			bgr_list = np.uint8(bgr_list)
# 			# 协方差矩阵
# 			conv_bgr = np.cov(np.float32(bgr_list.T))
# 			# 均值
# 			mean_bgr = np.mean(np.float32(bgr_list.T), axis=1)
# 			# 添加到字典
# 			conv_bgr_dict[color_name] = conv_bgr
# 			mean_bgr_dict[color_name] = mean_bgr
# 		# 构建颜色信息
# 		self.color_info  = {}
# 		self.color_info['cov'] = conv_bgr_dict
# 		self.color_info['mean'] = mean_bgr_dict
# 		# 对象序列化并保存
# 		with open(f'{self.config_folder}/color_block_statis.bin', 'wb') as f:
# 			pickle.dump(self.color_info, f)
	
# 	def load_color_info(self):
# 		'''载入颜色信息'''
# 		# 载入统计数据
# 		with open(f'{self.config_folder}/color_block_statis.bin', 'rb') as f:
# 			self.color_info = pickle.load(f)
# 		# 创建统计信息 多元正态分布
# 		self.multi_normal = {}
# 		for color_name in self.color_names:
# 			mean = self.color_info['mean'][color_name] # 均值
# 			cov = self.color_info['cov'][color_name] # 协方差矩阵
# 			self.multi_normal[color_name] = multivariate_normal(mean=mean, cov=cov)
	
# 	def img_bgr2binary(self, img_bgr, color_name):
# 		'''BGR图像转换为灰度值'''
# 		img_h, img_w = img_bgr.shape[:2]
# 		# 图像变形
# 		bgr_list = img_bgr.reshape(-1, 3)
# 		# 获取每个像素值的概率密度
# 		img_pdf_1d = self.multi_normal[color_name].pdf(bgr_list)
# 		# 使用概率密度进行二值化
# 		binary = np.uint8((img_pdf_1d.reshape(img_h, img_w) >= self.bin_pdf_threshold)) * 255
# 		# 数学形态学操作
# 		binary = cv2.erode(binary, np.ones((3, 3)))
# 		binary = cv2.dilate(binary, np.ones((9, 9)))
# 		return binary
# 	def is_legal_rect(self, rect):
# 		'''判断矩形框是否合法'''
# 		x, y, w, h = rect
# 		# 过滤掉小噪点
# 		if w < self.roi_min_size or h < self.roi_min_size:
# 			return False
# 		return True

# 	def color_clock_rect(self, img_bgr, color_name, canvas=None):
# 		'''获取色块的矩形区域'''
# 		# 创建画布
# 		if canvas is None:
# 			canvas = np.copy(img_bgr)
# 		rect_list = []
# 		# 图像二值化
# 		bianry = self.img_bgr2binary(img_bgr, color_name)
# 		# 连通域检测
# 		*_, contours, hierarchy = cv2.findContours(image=bianry,mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 		# 计算外接矩形
# 		for cnt in contours:
# 			# 计算外界矩形
# 			rect = cv2.boundingRect(cnt)
# 			# 判断是否合法
# 			if self.is_legal_rect(rect):
# 				# 添加外接矩形到列表
# 				rect_list.append(rect)
# 				# 绘制矩形框
# 				x, y, w, h = rect
# 				color = self.color_bgr[color_name]
# 				cv2.rectangle(canvas, (x, y), (x+w, y+h), color, thickness=4)

# 		return rect_list, canvas


# if __name__ == "__main__":
# 	# 自定义库
# 	from kyle_robot_toolbox.camera import USBCamera

# 	# 要识别的颜色列表
# 	# color_name_list = ["red", "green", "blue", "yellow"]
# 	color_name_list = ["red"]
# 	# 创建摄像头
# 	camera = USBCamera(config_path="config/usb_camera")
# 	# 初始相机
# 	camera.init_video_stream()
# 	# 载入标定数据
# 	camera.load_cam_calib_data()
# 	# 色块识别器
# 	color_block_detector = ColorBlockDetector()

# 	while True:
# 		# 开始时间
# 		t_start = time.time()
# 		img_bgr = camera.read_color_img()
# 		# 识别色块
# 		canvas = np.copy(img_bgr)
# 		for color_name in color_name_list:
# 			rect_list, canvas = color_block_detector.color_clock_rect(\
# 				img_bgr, color_name, canvas=canvas)
		
# 		# 统计帧率
# 		t_end = time.time()
# 		t_pass = t_end - t_start
# 		fps = int(1/t_pass)

# 		# 绘制帧率
# 		cv2.putText(canvas, text=f"FPS:{fps}",\
# 			org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
# 			fontScale=0.8, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
		
# 		cv2.imshow("canvas", canvas)
# 		key = cv2.waitKey(1)
# 		if key == ord('q'):
# 			break
# 	cv2.destroyAllWindows()
# 	camera.release()