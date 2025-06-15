try:
    from kyle_robot_toolbox.yolov8.yolov8_detect import YoloV8Detect
    from kyle_robot_toolbox.yolov8.yolov8_segment import YoloV8Segment
    __all__ = ['YoloV8Detect', 'YoloV8Segment']
except ModuleNotFoundError:
    __all__ = []