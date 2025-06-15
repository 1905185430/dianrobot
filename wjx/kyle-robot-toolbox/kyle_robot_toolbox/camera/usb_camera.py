'''
USB摄像头通用类
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import cv2
from kyle_robot_toolbox.camera import Camera

class USBCamera(Camera):
	'''USB摄像头'''

	def init_video_stream(self, cap_id=None):
		'''初始化视频流'''
		if cap_id is None:
			cap_id = int(self.config['device'])
		capture = cv2.VideoCapture(cap_id)
		# 设置分辨率
		# 设置图像高度
		capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config['img_height']))
		# 设置图像宽度
		capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config['img_width'])) 
		# 设置帧率
		capture.set(cv2.CAP_PROP_FPS,  int(self.config['fps']))
		# 设置编码方式	
		capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
		# 设置视频缓冲区
		capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config['buffer_size']))
		self.capture = capture
		return capture

	def read_color_img(self):
		'''获取彩色图'''
		# 通过OpenCV读取彩图
		ret, color_img = self.capture.read()
		return color_img
	
	def release(self):
		'''释放资源'''
		self.capture.release()