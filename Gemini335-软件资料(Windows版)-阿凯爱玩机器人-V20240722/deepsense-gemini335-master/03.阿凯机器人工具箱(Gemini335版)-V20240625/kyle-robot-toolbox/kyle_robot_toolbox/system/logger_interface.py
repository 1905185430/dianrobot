'''
日志工具库
----------------------------------------------------------------
作者: 阿凯爱玩机器人 | 微信: xingshunkai  | QQ: 244561792
官网: deepsenserobot.com
B站: https://space.bilibili.com/40344504
淘宝店铺: https://shop140985627.taobao.com
'''

import logging
from logging import getLogger

class LoggerInterface(object):
    '''日志接口'''
    def __init__(self, name=None, logging_level=None):
        # 日志基础配置
        logging.basicConfig(format='[%(name)s][%(levelname)s]: %(message)s')
        # 创建logger
        if name is None:
            # 设置name为类名
            name = self.__class__.__name__
        self.logger = getLogger(name)
        # 设置日志等级
        self.logging_level = logging_level
        if logging_level is None:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging_level)        
        