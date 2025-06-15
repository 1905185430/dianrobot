'''
四元数基本运算与插值算法
* lerp: 直线插值
* nlerp: 规范化直线插值
* slerp: 球面圆弧线性插值
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''

import numpy as np
from numpy import sin, cos, arcsin, arccos
# 导入numpy-quaternion库
import quaternion

# 自定义库
from .transform import *

class Quaternion:
	'''四元数'''
	def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
		'''初始化'''
		self.from_wxyz(w, x, y, z)
		
	def wxyz(self):
		'''以列表的形式返回四元数'''
		return np.float64([self.w, self.x, self.y, self.z])
	
	def xyzw(self):
		'''以列表的形式返回四元数'''
		return np.float64([self.x, self.y, self.z, self.w])
	
	def xyz(self):
		'''以列表的形式返回x, y, z'''
		return np.float64([self.x, self.y, self.z])

	@staticmethod
	def identity():
		'''单位四元数'''
		return Quaternion(1.0, 0.0, 0.0, 0.0)
	
	def copy(self):
		'''拷贝'''
		return Quaternion(self.w, self.x, self.y, self.z)
	
	def __str__(self) -> str:
		return "Quaternion[w={:.4f} x={:.4f} y={:.4f} z={:.4f}]".format(\
			float(self.w), float(self.x), float(self.y), float(self.z))
	
	def __add__(self,quaternion):
		'''四元数加法'''
		result=self.copy()
		result.w += quaternion.w
		result.x += quaternion.x
		result.y += quaternion.y
		result.z += quaternion.z
		return result
	
	def __sub__(self, quaternion):
		'''四元数减法'''
		result=self.copy()
		result.w -= quaternion.w
		result.x -= quaternion.x
		result.y -= quaternion.y
		result.z -= quaternion.z
		return result
	
	def __mul__(self, value):
		'''乘法'''
		if type(value) in [float, int, np.float_, np.int_]:
			# 四元数与标量相乘
			result = self.copy()
			result.w *= value
			result.x *= value
			result.y *= value
			result.z *= value
			return result
		elif type(value) == Quaternion:
			# 四元数之间的乘法
			return self.grabmann_product(value)
	
	def __rmul__(self, value):
		'''右乘'''
		return self.__mul__(value)
		
	def __truediv__(self, value):
		'''标量除法'''
		if type(value) in [float, int, np.float_, np.int_]:
			# 四元数与标量的除法
			result = self.copy()
			value_inv = 1.0 / value
			result.w *= value_inv
			result.x *= value_inv
			result.y *= value_inv
			result.z *= value_inv
			return result
		else:
			# 注: 不能直接除四元数
			return None
	
	def norm(self):
		'''获取四元数的模长/范数'''
		# return np.sqrt(self.w**2 + self.x**2 + self.y**2+ self.z**2)
		return quat_norm(self.wxyz())

	def unit(self):
		'''单位四元数(四元数规范化)'''
		q_unit = Quaternion()
		q_unit.from_wxyz(*quat_unit(self.wxyz()))
		return q_unit
	
	def is_unit(self):
		'''是否为单位四元数'''
		return self.norm() == 1.0
	
	def conjugate(self):
		'''四原数共轭'''
		result = self.copy()
		result.x *= -1
		result.y *= -1
		result.z *= -1
		return result
	
	def star(self):
		'''q^{*} 四元数共轭的别名'''
		return self.conjugate()
		
	def inverse(self):
		'''四元数求逆'''
		return self.conjugate() / self.norm()
	
	
	def reverse(self):
		'''四元数取反
		注: 对于旋转四元数 -q与q代表同一个旋转
		'''
		wxyz_reverse = [-1*v for v in self.wxyz()]
		return Quaternion(*wxyz_reverse)
	
	def grabmann_product(self, quaternion):
		'''四元数 Grabmann积'''
		q1_wxyz = self.wxyz()
		q2_wxyz = quaternion.wxyz()
		q_result = grabmann_product(q1_wxyz, q2_wxyz)
		return Quaternion(*q_result.reshape(-1))
	
	def dot(self, quaternion):
		'''点乘'''
		a, b, c, d = self.wxyz()
		e, f, g, h = quaternion.wxyz()
		return a*e + b*f + c*g + d*h
	
	def rotation(self, value):
		'''四元数旋转公式
		空间向量通过四元数进行旋转
		@value: 空间向量 list / 纯四元数
		'''
		# 旋转四元数
		q = self.copy()
		# 纯四元数 (代表空间向量)
		v = Quaternion()
		if type(value) in [list, np.ndarray]:
			# 判断是否是列表或ndarray的格式
			value = np.float64(value).reshape(-1)
			v.from_xyz(*value)
		elif type(value) == Quaternion:
			# 直接赋值
			v = value
		# 四元数旋转公式
		result = q*v*q.inverse()
		return result
	
	def angle_diff(self, quaternion, is_short_path=True):
		'''四元数之间的角度差(超球面)
		@quaternion: 四元数B
		@is_short_path: 选择长路径还是短路径
		 - True: theta in [0, pi/2]
		 - False: theta in [pi/2, pi]
		注: 3维刚体下对应的旋转角度为2*theta
		'''
		is_q_reverse = False
		# 将二者进行规范化
		q1_unit = self.unit()
		q2_unit = quaternion.unit()
		# 计算cos(theta)
		cos_theta = q1_unit.dot(q2_unit)
		# 因为双倍覆盖的原因 q跟-q在超球面相位差为pi
		# 但是本质上 q跟-q指代的是同一个旋转
		# 此时需要反转其中一个四元数
		if (is_short_path and cos_theta < 0) or (not is_short_path and cos_theta > 0):
			# is_short_path为真，强制约定了cos_theta > 0
			cos_theta *= -1
			is_q_reverse = True # 反转其中一个四元数
			# q2_unit = -1.0 * q2_unit
			# cos_theta = q1_unit.dot(q2_unit)
			
		# Arccos函数值域为[0, pi]
		# 当cos_theta > 0时, theta的取值范围为 [0, pi/2]
		# print(f"cos_theta={cos_theta}")
		# 约束下cos_theta的取值范围
		cos_theta = min(max(-1.0, cos_theta), 1.0)
		theta = np.arccos(cos_theta)
		# 四元数角度差theta， 对应刚体空间中的角度2*theta
		return theta, is_q_reverse
	
	def q_diff(self, q2):
		'''四元数旋转变化量
		q_1(自身)通过旋转得到了q_2
		delta_q * q_1 = q_2
		因此： delta_q = q_2 * (q_1)^{-1}
		当q1,q2都是单位四元数时
		delta_q = q_2 * (q_1)^{*}
		'''
		# 转换为单位向量
		q1 = self.copy().unit()
		q2 = q2.unit()
		delta_q = q2 * q1.star()
		return delta_q
	
	def interpolation_lerp(self, q2, n_segment=10, is_short_path=True, t_list=None):
		'''线性插值 在当前四元数与
		@q2: 四元数B
		@n_segment: 插值段
		@t_list: 时间序列 t \in [0, 1], 
			如果不指定t_list则按照n_segment等比例生成t_list
		'''
		n_segment = int(n_segment)
		# 计算四元数在两个超球面上的夹角
		theta, is_q_reverse = self.angle_diff(q2, is_short_path=is_short_path)
		if is_q_reverse:
			# 将q2取反
			q2 = q2.reverse()
		# 线性插值
		q_list = []
		q1 = self.copy()
		if t_list is None:
			t_list = [i_segment / n_segment  for i_segment in range(n_segment+1)]
		for t in t_list:
			qt = (1-t)*q1 + t*q2
			q_list.append(qt)
		return q_list, q2
	
	def interpolation_nlerp(self, q2, n_segment=10, is_short_path=True, t_list=None):
		'''正规划化线性插值 在当前四元数与
		@q2: 四元数B
		@n_segment: 插值段
		@t_list: 时间序列 t \in [0, 1], 
			如果不指定t_list则按照n_segment等比例生成t_list
		'''
		# 线性插值
		q_list, q2 = self.interpolation_lerp(q2, n_segment, \
			is_short_path=is_short_path, t_list=t_list)
		# 规范化
		return [q.unit() for q in q_list], q2
	
	def interpolation_slerp(self, q2, n_segment=10, is_short_path=True, t_list=None):
		'''球面线性插值
		@q2: 四元数B
		@n_segment: 插值段
		@t_list: 时间序列 t \in [0, 1], 
			如果不指定t_list则按照n_segment等比例生成t_list
		'''
		# 归一化
		q1 = self.unit()
		q2 = q2.unit()
		# print(f"q2_unit={q2}")
		# 检查夹角是否过小，如非常小则改用线性插值替换
		if q1.dot(q2) >= 0.9995:
			return self.interpolation_nlerp(q2, n_segment, \
				is_short_path=n_segment, t_list=t_list)
		# 计算四元数在两个超球面上的夹角
		theta, is_q_reverse = q1.angle_diff(q2, is_short_path=is_short_path)
		if is_q_reverse:
			# 将q2取反
			q2 = q2.reverse()
			# print(f"q2 reverse={q2}")
		# 插值
		q_list = []
		if t_list is None:
			# 生成等距的t_list
			t_list = [i_segment / n_segment  for i_segment in range(n_segment+1)]
		
		sin_theta_inv = 1.0 / sin(theta)
		# print(t_list)
		for t in t_list:
			# Slerp插值
			qt = sin_theta_inv * ((sin((1-t)*theta))*q1 + (sin(t*theta))*q2)
			# 规范化
			qt = qt.unit()
			# 追加到列表
			q_list.append(qt)
		return q_list, q2
	
	def from_wxyz(self, w, x, y, z):
		'''从wxyz的格式构造四元数'''
		self.w = w
		self.x = x
		self.y = y
		self.z = z
	
	def from_xyzw(self, x, y, z, w):
		'''从wxyz的格式构造四元数'''
		self.w = w
		self.x = x
		self.y = y
		self.z = z
  
	def from_xyz(self, x, y, z):
		'''从3D向量构造纯四元数'''
		self.w = 0
		self.x = x
		self.y = y
		self.z = z
	
	def from_euler(self, roll=0, pitch=0, yaw=0, unit="rad"):
		'''从欧拉角构造四元数
		注: 欧垃角为rpy格式
		'''
		if unit == "deg":
			roll, pitch, yaw = np.radians([roll, pitch, yaw],\
								 dtype=np.float64)
		qw, qx, qy, qz = euler2quat(roll, pitch, yaw)
		self.from_wxyz(qw, qx, qy, qz)
	
	def from_rmat(self, rmat):
		'''从旋转矩阵构造四元数
		参考链接: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
		'''
		qw, qx, qy, qz = rmat2quat(np.float64(rmat))
		self.from_wxyz(qw, qx, qy, qz)
		
	def from_rvect(self, u, theta=None):
		'''从旋转向量构造四元数
		@theta: 转轴
		@u: 转轴
		'''
		u = np.float64(u)
		# qw, qx, qy, qz = rvect2quat(u, theta=theta).astype(np.float64)
		qw, qx, qy, qz = rvect2quat(u, theta=theta).reshape(-1)
		self.from_wxyz(qw, qx, qy, qz)
	
	def to_euler(self):
		'''转换为欧垃角
		注: 欧垃角为rpy格式
		'''
		return quat2euler(self.wxyz())
	
	def to_rmat(self):
		'''转换为旋转矩阵'''
		return quat2rmat(self.wxyz())

	def to_rvect(self):
		'''转换为旋转向量'''
		return quat2rvect(self.wxyz())

	@staticmethod
	def mean(q_list, weights=None, 
			dtype=np.float32):
		'''
		四元数求均值
		[参数]
		@q_list (list / ndarray):
			四元数列表，可以是Quaternion列表， 也可以是N*4的ndarray矩阵
			注意传入的如果是ndarray, 列的顺序必须是w, x, y, z
		@weights:
			每个四元数所占的比重
		@dtype:
			返回的数据类型
		'''
		
		# 构造累加矩阵
		A = np.zeros((4, 4))
		# 获取四元数的个数
		q_num = len(q_list)
		
		if weights is None:
			# 构造同等权重的列表
			weight_mean = 1.0 / q_num
			weights = [weight_mean]*q_num
		# 权重累加值
		w_sum = 0
		# 累加
		for i in range(q_num):
			# 获得第i个四元数
			# 假定传入的四元数都是单位四元数
			q = q_list[i]
			# 如果q是Quaternion对象则需要将q转换为列向量
			if type(q) == Quaternion:
				q = np.float32(q.wxyz())
			else:
				# ndarray或者list对象
				q = np.float32(q).reshape(-1)
			# 四元数的权重
			w_i = weights[i]
			# q = [w, x, y, z]^T
			# 四元数向量外积 np.outer(q, q)
			# 外积后的结果, 是一个对称矩阵
			# [[w*w, w*x, w*y, w*z],
			#  [x*w, x*x, x*y, x*z],
			#  [y*w, y*x, y*y, y*z], 
			#  [z*w, z*x, z*y, z*z]]
			A += w_i * (np.outer(q, q))
			# 权重累加
			w_sum += w_i
		
		# 归一化
		A /= w_sum
		# 获得矩阵A的特征值最大值对应的特征向量作为四元数的均值
		q_mean = np.linalg.eigh(A)[1][:, -1]
		if dtype == Quaternion:
			# 转换为Quaternion对象
			q_mean_obj = Quaternion()
			q_mean_obj.from_wxyz(*q_mean)
			return q_mean_obj
		else:
			# 返回numpy ndarray格式的向量
			return q_mean
	
	@staticmethod
	def squad(q_waypoint, t_waypoint, n_segment=10, t_list=None):
		'''Squad插值
		@q_waypoint: 途经点四元数列表
		@t_waypoint: 途经点时间列表
		@t_list: 时序列表
		'''
		q_list = []
		# 自动生成t_list
		if t_list is None:
			t_list = [i_segment / n_segment  for i_segment in range(n_segment+1)]
		t_list = np.float32(t_list)

		# 将q_waypoint转换为wxyz列表
		wxyz_waypoint = np.array([np.quaternion(*q.wxyz()) for q in q_waypoint])
		t_waypoint = np.float32(t_waypoint)
		
		# squad插值
		
		wxyz_list = quaternion.squad(wxyz_waypoint, t_waypoint, t_list)
		wxyz_ndarray = quaternion.as_float_array(wxyz_list)
		# 在转换为四元数对象
		for wxyz in wxyz_ndarray:
			# 构造四元数对象
			q = Quaternion()
			q.from_wxyz(*wxyz)
			q_list.append(q)
		return q_list
	
	@staticmethod
	def get_z_list(q_list):
		'''获取Z轴列表(可视化用)'''
		# Z轴单位向量
		v_z_unit = np.float64([0, 0, 1])
		v_z_list = []
		for q in q_list:
			# 让Z轴发生旋转
			v_z_rotate = q.rotation(v_z_unit)
			v_z_list.append(v_z_rotate.xyz())
		v_z_list = np.float64(v_z_list)
		return v_z_list
