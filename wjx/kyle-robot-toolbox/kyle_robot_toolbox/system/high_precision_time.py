'''
高精度延时工具库
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''
import time

TIME_NS2S = 1.0 / (10**9)

def get_cur_time_s():
    '''获取高精度时间 单位s'''
    return time.time_ns() * TIME_NS2S

def sleep_s(delay_s):
	'''高精度延时'''
	t_target =  get_cur_time_s() + delay_s
	while get_cur_time_s() < t_target:
		pass

def speed_test(test_func, repeat_n=10000, is_debug=True):
	# 碰撞检测测速
	t_start = get_cur_time_s()
	for i in range(repeat_n):
		test_func()
		# is_collision = fcl_collide(convex_a, T_a, convex_b, T_b)
	t_end = get_cur_time_s()

	t_period = (t_end - t_start) / repeat_n
	if is_debug:
		print(f"周期: {t_period*1000.0:.5f}ms")
	return t_period