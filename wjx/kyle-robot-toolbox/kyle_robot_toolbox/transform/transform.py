'''
刚体空间变换 工具库
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np
from numpy import sin, cos, pi, arccos, arcsin, arctan, arctan2
import scipy
# 阿凯机器人工具箱
from kyle_robot_toolbox.math import *

def rzmat_2d(theta):
	'''旋转矩阵2D'''
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	T = [[cos_theta, -sin_theta, 0],
		 [sin_theta, cos_theta, 0],
		 [0, 0, 1.0]]
	return np.float64(T)


def tmat_2d(x, y, theta):
	'''2D空间变换'''
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	T = [[cos_theta, -sin_theta, x],
		[sin_theta, cos_theta, y],
		[0, 0, 1]]
	return np.float64(T)


def dmat(dx, dy, dz):
	'''沿着XYZ轴平移'''
	# T = np.eye(4)
	T = np_eye_4()
	T[0, 3] = np.float64(dx)
	T[1, 3] = np.float64(dy)
	T[2, 3] = np.float64(dz)
	return T


def dxmat(dx):
	'''沿X轴平移'''
	# T = np.eye(4)
	T = np_eye_4()
	T[0, 3] = np.float64(dx)
	return T


def dymat(dy):
	'''沿Y轴平移'''
	# T = np.eye(4)
	T = np_eye_4()
	T[1, 3] = np.float64(dy)
	return T


def dzmat(dz):
	'''沿Z轴平移'''
	# T = np.eye(4)
	T = np_eye_4()
	T[2, 3] = np.float64(dz)
	return T


def rxmat(gamma):
	'''绕X轴旋转'''
	gamma = np.float64(gamma)
	cos_gamma = cos(gamma)
	sin_gamma = sin(gamma)

	T = [[1.0, 0,           0,       0],
		 [0, cos_gamma, -sin(gamma), 0],
		 [0, sin(gamma),  cos_gamma, 0],
		 [0, 0,           0,         1.0]]
	return np.float64(T)


def rymat(beta):
	'''绕Y轴旋转'''
	beta = np.float64(beta)
	cos_beta = cos(beta)
	sin_beta = sin(beta)
	T = [[cos_beta,  0,  sin_beta, 0],
	 	 [0,       1.0,  0,        0],
		 [-sin_beta, 0,  cos_beta, 0],
		 [0,         0,  0,        1.0]]
	return np.float64(T)


def rzmat(alpha):
	'''绕Z轴旋转'''
	alpha = np.float64(alpha)
	cos_alpha = cos(alpha)
	sin_alpha = sin(alpha)
	T = [[cos_alpha, -sin_alpha, 0,   0],
		 [sin_alpha,  cos_alpha, 0,   0],
		 [0,           0,        1.0, 0],
		 [0,           0,          0, 1.0]]
	return np.float64(T)



def dhmat(alpha, a, theta, d):
	'''DH变换矩阵'''
	cos_alpha = cos(alpha)
	sin_alpha = sin(alpha)
	cos_theta = cos(theta)
	sin_theta = sin(theta)
	# V2版本
	T = [[cos_theta, -sin_theta, 0, a],
		 [sin_theta*cos_alpha, cos_alpha*cos_theta, -sin_alpha, -d*sin_alpha],
		 [sin_theta*sin_alpha, sin_alpha*cos_theta, cos_alpha, d*cos_alpha],
		 [0, 0, 0, 1.0]]
	return np.float64(T)
	
	# V1版本
	# dhmat = Transform.rxmat(alpha)
	# dhmat = dhmat.dot(Transform.dxmat(a))
	# dhmat = dhmat.dot(Transform.rzmat(theta))
	# dhmat = dhmat.dot(Transform.dzmat(d))
	# return dhmat


def euler2rmat(roll, pitch, yaw):
	'''欧拉角转换为旋转矩阵''' 
	alpha, beta, gamma = yaw, pitch, roll
	cos_gamma = np.cos(gamma)
	sin_gamma = np.sin(gamma)
	cos_beta = np.cos(beta)
	sin_beta = np.sin(beta)
	cos_alpha = np.cos(alpha)
	sin_alpha = np.sin(alpha)

	r11 = cos_alpha*cos_beta
	r12 = -sin_alpha*cos_gamma + sin_beta*sin_gamma*cos_alpha
	r13 = sin_alpha*sin_gamma + sin_beta*cos_alpha*cos_gamma
	r21 = sin_alpha*cos_beta
	r22 = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
	r23 = sin_alpha*sin_beta*cos_gamma - sin_gamma*cos_alpha
	r31 = -sin_beta
	r32 = sin_gamma*cos_beta
	r33 = cos_beta*cos_gamma
	R = [[r11, r12, r13],
		 [r21, r22, r23],
		 [r31, r32, r33]]
	return np.float64(R)


def rmat2euler(rmat):
	'''旋转矩阵转换为欧拉角'''
	# print(f"旋转矩阵: {rmat}")
	alpha = None # 偏航角
	beta = None  # 俯仰角
	gamma = None # 横滚角
	
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = np.copy(rmat).reshape(-1)
	if np.abs(r31) >= (1 - 0.000001):
		# 出现万向锁的问题
		if r31 < 0:
			gamma = 0
			beta = np.pi/2
			alpha = np.arctan2(r23, r22)
			return [[gamma, beta, alpha]]
		else:
			gamma = 0
			beta = -np.pi/2
			alpha = np.arctan2(-r23, r22)
			return [[gamma, beta, alpha]]
	else:
		# 正常求解
		cos_beta = np.sqrt(r32*r32 +r33*r33)
		cos_beta_list = [cos_beta, -cos_beta]
		rpy_list = []
		for cos_beta in cos_beta_list:
			if cos_beta == 0:
				# 存在cos_beta=0的情况
				continue
			beta = np.arctan2(-r31, cos_beta)
			alpha = np.arctan2(r21/cos_beta, r11/cos_beta)
			gamma = np.arctan2(r32/cos_beta, r33/cos_beta)
			rpy_list.append([gamma, beta, alpha])
		return rpy_list


def inverse(T):
	'''齐次变换矩阵求逆'''
	R = T[:3, :3]
	t = np.copy(T[:3, 3]).reshape((3, 1))
	R_T = R.T
	# T_inv = np.eye(4)
	T_inv = np_eye_4()
	T_inv[:3, :3] = R_T
	# 这里也是一样, reshape前需要拷贝一下
	# dot对于内存连续的数组速度会更快
	R_T = np.copy(R_T)
	t = np.copy(t)
	T_inv[:3, 3] = np.copy(-R_T.dot(t)).reshape(-1)
	return T_inv

# 
# def inverse(T):
# 	'''齐次变换矩阵求逆'''
# 	R = T[:3, :3]
# 	t = T[:3, 3].reshape((3, 1))
# 	R_T = R.T
# 	T_inv = np.eye(4)
# 	T_inv[:3, :3] = R_T
# 	T_inv[:3, 3] = -R_T.dot(t).reshape(-1)
# 	return T_inv


def skew(n):
	'''生成Skew矩阵'''
	xn, yn, zn = np.copy(n).reshape(-1)
	M = [[0, -zn, yn], \
		 [zn, 0, -xn], \
		 [-yn, xn, 0]]
	skew_mat = np.float64(M)
	return skew_mat

def rvect2rmat(u_unit, theta):
	'''旋转向量转换为旋转矩阵'''
	# 转轴归一化
	# u_unit = u / u_norm
	n1, n2, n3 = u_unit
	# 为了减少计算量，预先计算好
	a = np.sin(theta)
	b = 1 - np.cos(theta)
	an1 = a*n1
	an2 = a*n2
	an3 = a*n3
	bn1n1 = b*n1*n1
	bn2n2 = b*n2*n2
	bn3n3 = b*n3*n3
	bn1n2 = b*n1*n2
	bn2n3 = b*n2*n3
	bn1n3 = b*n1*n3
	# 计算旋转矩阵R
	R = [[1-bn2n2-bn3n3, -an3+bn1n2, an2+bn1n3],
		 [an3+bn1n2, 1-bn1n1-bn3n3, -an1+bn2n3],
		 [-an2+bn1n3, an1+bn2n3, 1-bn1n1-bn2n2]]
	R = np.float64(R)
	return R


def rvect2rmat2(u_unit, theta):
	'''旋转向量转换为旋转矩阵'''
	n1, n2, n3 = u_unit
	# 构造矩阵 n*n^{T}
	n1n2 = n1*n2
	n1n3 = n1*n3
	n2n3 = n2*n3
	mat_n_n_trans = np.float64([
		[n1**2, n1n2, n1n3],
		[n1n2, n2**2, n2n3],
		[n1n3, n2n3, n3**2]])
	
	# 构造矩阵 n_hat
	mat_n_hat = skew(u_unit)
	# 罗德里格斯公式
	# I = np.eye(3)
	I = np_eye_3()
	cos_theta = cos(theta)
	sin_theta = sin(theta)
	R = cos_theta*I + (1-cos_theta)*mat_n_n_trans + sin_theta*mat_n_hat
	return R


def rmat2rvect(rmat):
	'''旋转矩阵转换为旋转向量'''
	# 提取rmat中的元素 
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = np.copy(rmat).reshape(-1)
	# 计算旋转矩阵R的迹
	trace_rmat = r11 + r22 + r33
	# 根据迹的取值来分流 
	if trace_rmat >=3:
		# 单位矩阵
		# 转轴任意制定，旋转角度为0
		return np.float64([0, 0, 1]), np.float64(0)
	elif trace_rmat <= -1:
		# 绕X轴/Y轴/Z轴定轴旋转的情况
		# 转角为pi, 转轴为X, Y, Z基向量中的一个
		n = [1 if rii == 1 else 0 for rii in [r11, r22, r33]]
		return np.float64(n), np.pi
	else:
		# arccos(( trace(R) - 1 ) / 2)
		theta = np.arccos(0.5*(trace_rmat - 1))
		# 1 / ( 2 * sin(theta))
		ratio = 0.5 / np.sin(theta)
		# r32-r23是numpy的1d类型，必须手动转换为浮点数
		# n = ratio * np.float64([float(r32-r23), float(r13-r31), float(r21-r12)])
		n = ratio * np.float64([float(r32-r23), float(r13-r31), float(r21-r12)])
		return n, theta

def rmat2rvect2(rmat):
	'''旋转矩阵转换为旋转向量'''
	# 先将旋转矩阵转换为四元数
	rmat = np.float64(rmat)
	m11, m12, m13, m21, m22, m23, m31, m32, m33 = np.copy(rmat).reshape(-1)
	# trace是矩阵的迹, 是矩阵主对角元素之和
	# trace(rmat) = m11 + m22 + m33
	trace_rmat = m11 + m22 + m33
	if  trace_rmat > 0:
		# 注: q0不能是0， 否则就变成了纯四元数了
		# 就不是旋转四元数了
		# S = 4q0
		s = np.sqrt(trace_rmat+1) * 2.0 # S = 4*qw
		inv_s = 1.0 / s
		qw = 0.25 * s
		qx = (m32 - m23) * inv_s
		qy = (m13 - m31) * inv_s
		qz = (m21 - m12) * inv_s
	elif m11 > m22 and m11 > m33:
		s = np.sqrt(1.0 + m11 - m22 - m33) * 2 # S = 4*qx
		inv_s = 1.0 / s
		qw = (m32 - m23) * inv_s
		qx = 0.25 * s
		qy = (m12 + m21) * inv_s
		qz = (m13 + m31) * inv_s
	elif m22 > m33:
		s = np.sqrt(1.0 - m11 + m22 - m33) * 2 # S = 4*qy
		inv_s = 1.0 / s
		qw = (m13 - m31) * inv_s
		qx = (m12 + m21) * inv_s
		qy = 0.25 * s
		qz = (m23 + m32) * inv_s
	else:
		s = np.sqrt(1.0 - m11 - m22 + m33) * 2 # S = 4*qz
		inv_s = 1.0 / s
		qw = (m21 - m12)
		qx = (m13 + m31)
		qy = (m23 + m32)
		qz = 0.25 * s
	# 角度的1/2
	theta_d2 = np.arccos(qw)
	theta = theta_d2 * 2
	sin_theta_d2 = sin(theta_d2)
	ux = qx / sin_theta_d2
	uy = qy / sin_theta_d2
	uz = qz / sin_theta_d2
	U = [ux, uy, uz]
	return np.float64(U), theta

def mrp2rvect(N):
	'''修正罗德里格斯公式转换为旋转向量'''
	# 求解theta
	# theta = np.arcsin(0.5*math_norm(N))*2
	theta = np.arcsin(0.5*np.linalg.norm(N))*2
	n = N / (2*np.sin(0.5*theta))
	return n, theta


def rvect2mrp(n, theta):
	'''旋转向量转换为修正罗德里格斯参数'''
	N = 2*np.sin(0.5*theta)*n
	return N

def mrp2rmat(N):
	'''修正罗德里格斯参数转换为旋转矩阵'''
	# N的模长
	N_norm = np.linalg.norm(N) # math_norm(N)
	# N的模长平方
	N_norm_pw2 = N_norm*N_norm
	# 修正罗德里格斯公式
	# I = np.eye(3)
	I = np_eye_3()
	R = (1-0.5*N_norm_pw2)*I + 0.5*(N*N.T + \
		np.sqrt(4-N_norm_pw2)*skew(N))
	return R


def rmat2mrp(rmat):
	'''旋转矩阵转换为修正罗德里格斯参数'''
	# 旋转矩阵转换为旋转向量
	n, theta = rmat2rvect(rmat)
	# 通过旋转向量构造旋转矩阵
	N = 2*np.sin(0.5*theta)*n
	return N

# TODO 新增功能

def vect_norm(v_xyz):
	'''向量模长'''
	x, y, z = v_xyz.astype(np.float64)
	return np.sqrt(x**2 + y**2 + z**2)


def quat_norm(q_wxyz):
	'''四元数模数'''
	# 维度需要手动转换
	qw, qx, qy, qz = q_wxyz.astype(np.float64)
	# qw, qx, qy, qz = np.float64(q_wxyz)
	# 闭坑指南，np.sqrt得到的是numpy的一维数组
	return np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)

def rmat2quat(rmat):
	'''旋转矩阵转换为四元数'''
	# 参考链接: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
	m11, m12, m13, m21, m22, m23, m31, m32, m33 = np.copy(rmat).reshape(-1)
	# trace是矩阵的迹, 是矩阵主对角元素之和
	# trace(rmat) = m11 + m22 + m33
	trace_rmat = m11 + m22 + m33
	if  trace_rmat > 0:
		# 注: q0不能是0， 否则就变成了纯四元数了
		# 就不是旋转四元数了
		# S = 4q0
		s = np.sqrt(trace_rmat+1) * 2.0 # S = 4*qw
		inv_s = 1.0 / s
		qw = 0.25 * s
		qx = (m32 - m23) * inv_s
		qy = (m13 - m31) * inv_s
		qz = (m21 - m12) * inv_s
	elif m11 > m22 and m11 > m33:
		s = np.sqrt(1.0 + m11 - m22 - m33) * 2 # S = 4*qx
		inv_s = 1.0 / s
		qw = (m32 - m23) * inv_s
		qx = 0.25 * s
		qy = (m12 + m21) * inv_s
		qz = (m13 + m31) * inv_s
	elif m22 > m33:
		s = np.sqrt(1.0 - m11 + m22 - m33) * 2 # S = 4*qy
		inv_s = 1.0 / s
		qw = (m13 - m31) * inv_s
		qx = (m12 + m21) * inv_s
		qy = 0.25 * s
		qz = (m23 + m32) * inv_s
	else:
		s = np.sqrt(1.0 - m11 - m22 + m33) * 2 # S = 4*qz
		inv_s = 1.0 / s
		qw = (m21 - m12)
		qx = (m13 + m31)
		qy = (m23 + m32)
		qz = 0.25 * s
	q_wxyz = [qw, qx, qy, qz]
	return np.float64(q_wxyz)

def rvect2quat(u, theta=None):
	'''从旋转向量构造四元数
	@theta: 转轴
	@u: 转轴
	'''
	# 旋转轴
	# u = np.float64(u).reshape(-1)
	u = u.astype(np.float64)

	# 旋转向量 旋转轴归一化
	u_norm = vect_norm(u) # math_norm(u)
	# 如果没有指定norm则用转轴模长作为角度
	if theta is None:
		theta = u_norm
	if u_norm != 1.0:
		u /= u_norm
	# 根据四元数的旋转公式
	# 实质是旋转两次， 因此单次旋转角度为
	# 刚体旋转角度的一半
	# q_theta = theta / 2.0
	# TODO 这里有问题
	q_theta = 0.5 * theta
	qw = cos(q_theta)
	qx, qy, qz = (sin(q_theta) * u).astype(np.float64)
	# 这里有个问题qw, qx, qy, qz 都是numpy的类型的数值
	q_wxyz = [float(qw), float(qx), float(qy), float(qz)]
	return np.float64(q_wxyz)



def quat_unit(q_wxyz):
	'''四元数转换为单位四元数'''
	# q_wxyz = np.float64(q_wxyz)
	# 要改用.astype(np.float64)
	q_wxyz = q_wxyz.astype(np.float64) 
	return q_wxyz / quat_norm(q_wxyz)


def quat2rmat(q_wxyz):
	'''四元数转换为旋转矩阵'''
	q_wxyz = quat_unit(q_wxyz)
	q0, q1, q2, q3 = q_wxyz
	q1_pw2_2 = 2.0*q1*q1
	q2_pw2_2 = 2.0*q2*q2
	q3_pw2_2 = 2.0*q3*q3
	q0_q1_2 = 2.0*q0*q1
	q0_q2_2 = 2.0*q0*q2
	q0_q3_2 = 2.0*q0*q3
	q1_q2_2 = 2.0*q1*q2
	q2_q3_2 = 2.0*q2*q3
	q1_q3_2 = 2.0*q1*q3

	R = [[1.0-q2_pw2_2-q3_pw2_2, q1_q2_2-q0_q3_2, q1_q3_2+q0_q2_2],
		 [q1_q2_2+q0_q3_2, 1.0-q1_pw2_2-q3_pw2_2, q2_q3_2-q0_q1_2],
		 [q1_q3_2-q0_q2_2, q2_q3_2+q0_q1_2, 1.0-q1_pw2_2-q2_pw2_2]]
	return np.float64(R)


def quat2rvect(q_wxyz):
	'''四元数转换为旋转向量
	'''
	# 四元数转换为单位四元数
	w, x, y, z = quat_unit(q_wxyz)
	if w > 0.9995:
		return np.float64([0, 0, 1]), 0 
	# 角度的1/2
	theta_d2 = arccos(w)
	theta = theta_d2 * 2
	sin_theta_d2 = sin(theta_d2)
	ux = x / sin_theta_d2
	uy = y / sin_theta_d2
	uz = z / sin_theta_d2
	U = [ux, uy, uz]
	return np.float64(U), theta

# 
def euler2quat(roll=0, pitch=0, yaw=0):
	'''欧拉角转换为四元数'''
	# 欧拉角转换为旋转矩阵
	rmat = euler2rmat(roll=roll, pitch=pitch, yaw=yaw)
	# 旋转矩阵转换为四元数
	return rmat2quat(rmat)


def quat2euler(q_wxyz):
	'''四元数转换为欧拉角'''
	# 四元数转换为旋转矩阵
	rmat = quat2rmat(q_wxyz)
	# 旋转矩阵转换为欧拉角
	return rmat2euler(rmat)


def grabmann_product(q1_wxyz, q2_wxyz):
	'''四元数 Grabmann积'''
	a, b, c, d = q1_wxyz.astype(np.float64) # np.float64(q1_wxyz)
	# 四元数相乘矩阵(左乘)
	mat_left = [[a, -b, -c, -d],
				[b, a, -d, c],
				[c, d, a, -b],
				[d, -c, b, a]]
	mat_left = np.float64(mat_left)
	q2_vect = np.copy(q2_wxyz).reshape((-1, 1))
	# q_result = np.copy(np.dot(mat_left, q2_vect)).reshape(-1)
	q_result = np.copy(np.dot(mat_left, q2_vect)).astype(np.float64)
	return q_result



class Transform:
	@staticmethod
	def rzmat_2d(theta):
		'''旋转矩阵2D'''
		return rzmat_2d(theta)
	
	@staticmethod
	def tmat_2d(x, y, theta):
		'''2D空间变换'''
		return tmat_2d(x, y, theta)
	
	@staticmethod
	def dmat(dx, dy, dz):
		'''沿着XYZ轴平移'''
		return dmat(dx, dy, dz)

	@staticmethod
	def dxmat(dx):
		'''沿X轴平移'''
		return dxmat(dx)
	
	@staticmethod
	def dymat(dy):
		'''沿Y轴平移'''
		return dymat(dy)
	
	@staticmethod
	def dzmat(dz):
		'''沿Z轴平移'''
		return dzmat(dz)
	
	@staticmethod
	def rxmat(gamma, ndim=4):
		'''绕X轴旋转'''
		rmat = rxmat(gamma)
		if ndim == 4:
			return rmat
		else:
			return rmat[:3, :3]

	@staticmethod
	def rymat(beta, ndim=4):
		'''绕Y轴旋转'''
		rmat = rymat(beta)
		if ndim == 4:
			return rmat
		else:
			return rmat[:3, :3]
		
	@staticmethod
	def rzmat(alpha, ndim=4):
		rmat = rzmat(alpha)
		if ndim == 4:
			return rmat
		else:
			return rmat[:3, :3]
	
	@staticmethod
	def dhmat(alpha, a, theta, d):
		'''DH变换矩阵'''
		return dhmat(alpha, a, theta, d)

	@staticmethod
	def euler2rmat(roll=0, pitch=0, yaw=0, dim=3):
		'''欧拉角转换为旋转矩阵''' 
		rmat = euler2rmat(roll=roll, pitch=pitch, yaw=yaw)
		if dim == 4:
			# rmat2 = np.eye(4)
			rmat2 = np_eye_4()
			rmat2[:3, :3] = rmat
			return rmat2
		else:
			return rmat
	
	@staticmethod
	def rmat2euler(rmat):
		'''旋转矩阵转换为欧拉角'''
		return rmat2euler(rmat)
	
	@staticmethod
	def inverse(T):
		'''齐次变换矩阵求逆'''
		return inverse(T)
	
	@staticmethod
	def skew(n):
		'''生成Skew矩阵'''
		n = np.float64(n).reshape(-1)
		return skew(n)
	
	@staticmethod
	def rvect2rmat(u, theta=None):
		'''旋转向量转换为旋转矩阵'''
		# 转轴
		u = np.float64(u).reshape(-1)
		# 转轴模长
		u_norm = math_norm(u)
		# 转轴模长
		u_norm = math_norm(u)
		if theta is None:
			# 如果没有指定theta则将旋转向量模长作为角度
			theta = u_norm
		# 转轴归一化
		u_unit = u / u_norm
		return rvect2rmat(u_unit, theta)
	
	@staticmethod
	def rvect2rmat2(u, theta=None):
		'''旋转向量转换为旋转矩阵'''
		# 转轴
		u = np.float64(u).reshape(-1)
		# 转轴模长
		u_norm = math_norm(u)
		# 转轴模长
		u_norm = math_norm(u)
		if theta is None:
			# 如果没有指定theta则将旋转向量模长作为角度
			theta = u_norm
		# 转轴归一化
		u_unit = u / u_norm
		return rvect2rmat2(u_unit, u_unit)
	
	@staticmethod
	def rmat2rvect(rmat):
		'''旋转矩阵转换为旋转向量'''
		return rmat2rvect(rmat)
	
	@staticmethod
	def rmat2rvect2(rmat):
		'''旋转矩阵转换为旋转向量'''
		return rmat2rvect2(rmat)
	
	@staticmethod
	def mrp2rvect(N):
		'''修正罗德里格斯公式转换为旋转向量'''
		return mrp2rvect(N)
	
	@staticmethod
	def rvect2mrp(n, theta):
		'''旋转向量转换为修正罗德里格斯参数'''
		return rvect2mrp(n, theta)
		
	@staticmethod
	def mrp2rmat(N):
		'''修正罗德里格斯参数转换为旋转矩阵'''
		N = np.float64(N)
		# 强制将N变为列向量
		N = np.copy(N).reshape((-1, 1))

		return mrp2rmat(N)
	
	@staticmethod
	def rmat2mrp(rmat):
		'''旋转矩阵转换为修正罗德里格斯参数'''
		return rmat2mrp(rmat)

	@staticmethod
	def rmat2quat(rmat):
		'''旋转矩阵转换为四元数'''
		return rmat2quat(rmat)
		
	@staticmethod
	def rvect2quat(u, theta=None):
		'''旋转向量转换为四元数'''
		return rvect2quat(u, theta=theta)

	@staticmethod
	def quat2rmat(q_wxyz):
		'''四元数转换为旋转矩阵'''
		return quat2rmat(q_wxyz)
	
	@staticmethod
	def quat2rvect(q_wxyz):
		'''四元数转换为旋转向量
		'''
		return quat2rvect(q_wxyz)
	
	@staticmethod
	def euler2quat(roll=0, pitch=0, yaw=0):
		'''欧拉角转换为四元数'''
		return euler2quat(roll, pitch, yaw)
	
	@staticmethod
	def quat2euler(q_wxyz):
		'''四元数转换为欧拉角'''
		return quat2euler(q_wxyz)
	
	@staticmethod
	def grabmann_product(q1_wxyz, q2_wxyz):
		'''四元数 Grabmann积'''
		return grabmann_product(q1_wxyz, q2_wxyz)
	
	@staticmethod
	def position_linear_interpolation(p_waypoint, t_waypoint, n_segment=10, t_list=None):
		'''位置直线插值'''
		p_waypoint = np.float32(p_waypoint)
		t_waypoint = np.float32(t_waypoint)
		
		# 自动生成t_list
		if t_list is None:
			t_list = [i_segment / n_segment  for i_segment in range(n_segment+1)]
		t_list = np.float32(t_list)

		# 在XYZ三个轴上，分别进行插值
		p_list = np.zeros((len(t_list), 3))
		# 先进行线性插值
		for i in range(3):
			waypoint_dim_i = p_waypoint[:, i]
			f_linear = scipy.interpolate.interp1d(t_waypoint, waypoint_dim_i)
			p_list[:, i] = f_linear(t_list)
		return p_list
