mport time
import sys
import os

# 导入阿凯写的Orbbec工具库
# 确保orbbec_utils.py跟你目前所执行的脚本在同一级目录下
from orbbecsdk_utils import *
# 添加Python Path
add_path_pyorbbecsdk()

# 导入pyorbbecsdk
from pyorbbecsdk import OBFormat, OBError,Pipeline,Config,OBSensorType,OBFormat,OBError # 明确导入 OBFormat
from pyorbbecsdk import *
