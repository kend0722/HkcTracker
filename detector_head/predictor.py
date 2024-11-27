#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 15:07
@Description: predictor - 算法检测头，帧图像->图像结果
@Modify:
@Contact: tankang0722@gmail.com

"""
import torch
import cv2
import numpy as np
import os.path as osp
from data.dataset.preprocess import preprocess_image
from track_utils.post_process.post_processing import postprocess

"""
作用：
    初始化模型和其他必要的属性。
    处理TensorRT模型（可选）。如果是边缘设备建议使用onnx或者openvino。
    设置图像预处理参数。
    提供 inference 方法，用于图像预处理、模型推理和后处理。
    返回推理结果和图像信息。
"""

class Predictor(object):
    """这是一个用于预测被检测物体的类，主要负责图像预处理、模型推理和后处理。"""
    # 初始化属性,trt_file,TensorRT模型文件路径，默认为 None。decoder: 解码函数，默认为 None。
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        # 处理TensorRT模型
        if trt_file is not None:  # 如果提供了TensorRT模型文件路径。
            from torch2trt import TRTModule
            # 创建TensorRT模块对象。
            model_trt = TRTModule()
            # 加载TensorRT模型的状态字典。
            model_trt.load_state_dict(torch.load(trt_file))
            # 创建一个全1的张量，进行一次前向推理，确保模型准备好。
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt  # 将模型替换为TensorRT模型。
        # 设置图像预处理参数，当然这里知识为了保证视频帧的所有图像色彩一致性
        self.rgb_means = (0.485, 0.456, 0.406)  # 设置RGB通道的均值。
        self.std = (0.229, 0.224, 0.225)    # 设置RGB通道的标准差。


    """ 推理方法 inference """
    def inference(self, img, timer):

        """ 步骤1 给每帧图像顺序处理记录id """
        # 初始化图像信息字典。
        img_info = {"id": 0}    # 初始化图像信息字典。
        # 处理图像路径
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)   # 获取图像文件名。
            img = cv2.imread(img)
        else:
            #  如果输入不是一个字符串。
            img_info["file_name"] = None    # 文件名设为 None。

        # 获取图像尺寸
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        """ 图像预处理 """
        img, ratio = preprocess_image(img, self.test_size, self.rgb_means, self.std)  # 对图像进行预处理，包括缩放、归一化等。
        img_info["ratio"] = ratio   # 存储缩放比例。
        # 将图像转换为张量，并添加批次维度
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        """ 步骤2: 前向传播 推理 *note: 可以在这里替换为你的目标检测的转换头 """
        # 如果启用了半精度推理，将图像转换为半精度浮点数。需要与模型的精度保持一致
        if self.fp16:
            img = img.half()  # to FP16
        # 模型推理，前向传播
        with torch.no_grad():
            timer.start()  # 开始计时。
            outputs = self.model(img)   # 进行模型推理。替换为yolov5推理。v5版本的人体检测
            if self.decoder is not None:    # 如果有解码函数。
                outputs = self.decoder(outputs, dtype=outputs.type())   # 对输出进行解码。
            # 对输出进行后处理，包括NMS等。
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            print(timer.stop(average=False), 'inference')
        """
           outputs:list()
                   output = [
                        detections_image_N -> torch.tensor([
                        [10.0, 20.0, 50.0, 60.0, 0.9, 0.8, 1.0],  # 第n个图像的第一个检测框
                        ...
                    ]   
           img_info    # 返回推理结果和图像信息。 out
        """
        return outputs, img_info    # 返回推理结果和图像信息。支持多张的图像
