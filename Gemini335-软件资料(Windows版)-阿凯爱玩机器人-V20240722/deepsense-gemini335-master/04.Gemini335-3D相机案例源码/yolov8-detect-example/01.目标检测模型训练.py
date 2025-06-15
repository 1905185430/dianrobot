#!/usr/bin/env python
# coding: utf-8

# ## 导入依赖

# In[3]:


import os
from ultralytics import YOLO


# ## 创建一个YoloV8对象

# In[5]:


# 模型路径
model_config_path = os.path.join(os.path.abspath("."), "yolov8n.yaml")
print(model_config_path)


# In[6]:


# 创建一个YoloV8n的模型
# 注意事项: 传入绝对路径
# 传入的是yolov8n.yaml, YOLO会找到yolov8.yaml并使用里面的n选项
model = YOLO(model_config_path)


# ## 训练模型

# In[7]:


# 模型路径
dataset_config_path = os.path.join(os.path.abspath("."), "..", "datasets", "bottle_cap.yaml")
print(dataset_config_path)


# In[ ]:


# 这里得传入数据集YAML配置文件绝对路径
# 注意事项: 
# 1.不可传入相对路径
# 2.imgsz需要是32的倍数
model.train(data=dataset_config_path, batch=10, epochs=300, imgsz=1088)

