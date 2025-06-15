import os
from ultralytics import YOLO

## 创建一个YoloV8实例分割模型
model_config_path = os.path.join(os.path.abspath("."), "yolov8-seg.yaml")
model = YOLO(model_config_path)

## 训练模型
# 注: batch不能大，英伟达1080Ti 12G的显卡内存有点不够用。 容易报错。 
# 这里得传入数据集YAML配置文件绝对路径
# 数据集配置文件
dataset_config_path = os.path.join(os.path.abspath("."), "..", "datasets", "color_block.yaml")
model.train(data=dataset_config_path, batch=4, epochs=1000, imgsz=1088)

