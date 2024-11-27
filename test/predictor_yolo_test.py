#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 15:23
@Description: predictor_yolo_test.py - 测试yolov5推理模型类
@Modify:
@Contact: tankang0722@gmail.com
"""

import os
import sys
import cv2
from detector_head.yolov5_predictor import Yolov5Predictor
from track_utils.my_timer import MyTimer
import importlib.util


# # 添加模块所在目录到 sys.path
# sys.path.append(os.path.abspath('/'))
# # 打印 sys.path 来验证是否添加成功
#
#
# # 检查 utils 目录和 timer.py 文件是否存在
# utils_path = os.path.join('/', 'utils')
# timer_path = os.path.join(utils_path, 'my_timer.py')
#
#
# # 尝试动态导入模块
# try:
#     spec = importlib.util.spec_from_file_location("my_timer", timer_path)
#     timer_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(timer_module)
#     MyTimer = timer_module.MyTimer
#     print("Imported Timer successfully")
# except ModuleNotFoundError as e:
#     print(f"Module not found: {e}")
# except Exception as e:
#     print(f"Other error: {e}")


if __name__ == '__main__':
    timer = MyTimer()
    timer.start()
    image_path = r"D:\kend\other\test01.jpg"
    model_path = r"D:\kend\other\yolov5n.pt"
    predictor = Yolov5Predictor(model_path=model_path)
    person_result = predictor.predict(image=cv2.imread(image_path))
    print(person_result)
    timer.stop()
    print(f"耗时：{timer.duration}s")
