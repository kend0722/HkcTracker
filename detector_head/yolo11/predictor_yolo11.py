# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/27 下午8:34
@Author  : Kend
@FileName: predictor_yolo11.py
@Software: PyCharm
@modifier:
"""
from ultralytics import YOLO
import cv2
import numpy as np




class PredictorYolo11:
    def __init__(self, model_path, input_size=(640, 640)):
        self.model = YOLO(model_path)
        self.input_size = input_size

    def predict(self, image: np.ndarray, timer=None):
        timer.start()
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image
        img_info["ratio"] = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        tensorboard_data = None
        # --------推理---------
        results_list = self.model.predict(
            source=image,
            task='detect',
            conf=0.1
        )
        for results in results_list:  # 遍历检测结果,这是一个ultralytics.engine.results.Results对象
            # tensorboard_data = results.boxes.numpy().data  # 推理的原始张量的numpy形式的data数据
            tensorboard_data = results.boxes.data  # 推理的原始张量的numpy形式的data数据
        return tensorboard_data, img_info


if __name__ == '__main__':
    image = r"D:\kend\myPython\ultralytics-main\ultralytics\assets\bus.jpg"
    model_path = r"D:\kend\myPython\Hk_Tracker\detector_head\yolo11\yolo11s.onnx"
    img = cv2.imread(image)
    predictor_yolo11 = PredictorYolo11(model_path)
    resource, info = predictor_yolo11.predict(img)
    print(resource, info)

