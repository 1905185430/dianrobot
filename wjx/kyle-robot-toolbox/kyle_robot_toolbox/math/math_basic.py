'''
数学运算-基础计算
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import math
import numpy as np


def math_norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def np_eye_3():
    return np.eye(3)

def np_eye_4():
    return np.eye(4)

# 单位矩阵
I = np_eye_4()