from kyle_robot_toolbox.camera.camera import Camera
from kyle_robot_toolbox.camera.usb_camera import USBCamera

__all__ = ['Camera', 'USBCamera']

# 奥比中光Gemini2
try:
    from kyle_robot_toolbox.camera.gemini335 import Gemini335
    __all__.append('Gemini335')
except ImportError as e:
    # DLL加载有问题
    print("[ERROR] 奥比中光pyorbbecsdk动态链接库加载出错")
    print(e)
    pass
except ModuleNotFoundError as e:
    # 没有配置pyorbbecsdk
    print("[WARN] 没有安装奥比中光pyorbbecsdk")
    pass

