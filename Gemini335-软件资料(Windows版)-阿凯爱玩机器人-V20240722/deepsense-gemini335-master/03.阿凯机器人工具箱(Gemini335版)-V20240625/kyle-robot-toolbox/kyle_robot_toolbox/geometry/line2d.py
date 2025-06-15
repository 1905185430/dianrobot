'''
计算几何工具包 2D直线
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import numpy as np

def distance_point2segment(A, B, P):
    '''计算点P到线段AB的距离'''
    # 转换为列向量
    A = np.float32(A).reshape((-1, 1))
    B = np.float32(B).reshape((-1, 1))
    P = np.float32(P).reshape((-1, 1))
    
    AP = P - A
    AB = B - A
    alpha = np.dot(AP.reshape(-1), AB.reshape(-1)) / np.linalg.norm(AB)**2  
    
    if alpha < 0:
        # AC在AB的相反向量, 此时线段上距离P点最近为A点
        return np.linalg.norm(AP)
    elif alpha <= 1:
        # 点C在线段AB上, 最近的距离为PC
        AC = alpha * AB
        PC = AC - AP
        return np.linalg.norm(PC)
    else:
        # C点超出线段之外,B点距离P点最近
        PB = AB - AP
        return np.linalg.norm(PB)

def line_point2kb(A, B):
    '''根据直线上的两个点的坐标, 求k,b'''
    xa, ya = A
    xb, yb = B
    if xa == xb:
        xa += 1e-3
    k = (ya - yb) / (xa - xb)
    b = ya - k*xa
    return k, b

def line_cross_pt(k1, b1, k2, b2):
    '''计算两条直线的交点'''
    A = np.float32([
        [-k1, 1],
        [-k2, 1]])
    if np.linalg.det(A) == 0:
        # 行列式为0说明两条直线平行,没有交点/两条直线重合
        return False, None
    
    b = np.float32([b1, b2]).reshape((-1, 1))
    cx, cy = (np.linalg.inv(A).dot(b)).reshape(-1)
    return True, (int(cx), int(cy))

def line_cross_pt2(L1_A, L1_B, L2_A, L2_B):
    '''计算两条直线的交点'''
    # 分别计算两条直线的表达式
    L1_AB_k, L1_AB_b = line_point2kb(L1_A, L1_B)
    L2_AB_k, L2_AB_b = line_point2kb(L2_A, L2_B)
    
    return line_cross_pt(L1_AB_k, L1_AB_b, L2_AB_k, L2_AB_b)

def is_point_in_convex(convex_pts, p):
    '''判断点是否在凸包内(凸包顺序为顺指针)'''
    n_convex_pt = len(convex_pts)
    p = np.float32(list(p)).reshape((-1, 1))
    for i in range(n_convex_pt):
        a = np.float32(list(convex_pts[i])).reshape((-1, 1))
        b = np.float32(list(convex_pts[(i+1)%n_convex_pt])).reshape((-1, 1))
        vi = b - a
        vp = p - a
        if np.cross(vp.reshape(-1), vi.reshape(-1)) < 0:
            return False
    return True