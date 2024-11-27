#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 15:12
@Description: preprocess - 数据预处理
@Modify:
@Contact: tankang0722@gmail.com
"""
import numpy as np
import cv2


def preprocess_image(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
    """
    预处理图像
    image: 输入的原始图像数据。
    input_size: 模型要求的输入尺寸，是一个二维元组 (height, width)。
    mean: 图像归一化使用的均值，可以是单个值或一个列表/元组，对应每个通道的均值。
    std: 图像归一化使用的标准差，可以是单个值或一个列表/元组，对应每个通道的标准差。
    swap: 图像通道顺序的调整，默认值为 (2, 0, 1)，意味着从 (height, width, channels) 转换到 (channels, height, width)。
    Return : 返回处理好的图像以及计算出的缩放比例 r，r后者可用于后续的结果恢复，例如检测框的坐标调整.
    """
    # 创建填充图像：
    # rgb通道填充114.0，灰度图填充114.0
    # cv2.imshow("img1", image)
    # cv2.waitKey(0)
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0

    # 调整图像大小并填充到指定尺寸：
    img = np.array(image)
    # print(img.shape)  # (720, 1280, 3)  原图
    # cv2.imshow("img1", padded_img)
    # cv2.waitKey(0)
    # 计算缩放比例
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])  # 计算原图最长边到binding box的比例
    # print(r) # 0.5
    # TODO : 缩放有问题  -> 是因为np.float32的问题 我们将像素值归一化到0到1之间，这样图像才能正确显示。还需要居中保持
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR, ).astype(np.float32)
    # resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
    #                            interpolation=cv2.INTER_AREA).astype(np.uint8)
    # 将调整后的图像放置到填充图像中：
    # 计算居中的位置
    # x_offset = (input_size[1] - int(img.shape[1] * r)) // 2  # 宽度
    # y_offset = (input_size[0] - int(img.shape[0] * r)) // 2   # 高度
    # # 将缩放后的图像粘贴到新的空白图像上
    # padded_img[y_offset:y_offset + int(img.shape[0] * r),  x_offset:x_offset +
    #                                                       int(img.shape[1] * r)] = resized_img
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img  # 没有居中
    # print(padded_img.shape) # (640, 640, 3)
    # padded_img /= 255.0
    # cv2.imshow("resized_img", padded_img)
    # cv2.waitKey(0)
    # 转置 BGR 2 RGB
    # im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    padded_img = padded_img[:, :, ::-1]
    # 归一化
    padded_img /= 255.0
    #  根据提供的均值和标准差对图像进行进一步的归一化处理，最好与目标检测的训练参数保持一致。# TODO : yolov5 默认是NONE
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    # 调整通道顺序
    padded_img = padded_img.transpose(swap)
    # 确保数据在内存中是连续存储的
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # 返回预处理后的图像和缩放比例
    return padded_img, r



if __name__ == '__main__':
    img_path = r"D:\kend\work\Hk_Tracker\data\dataset\test_images\frame_0000.jpg"
    image = cv2.imread(img_path)
    preprocess_image(image, input_size=(1280, 1280))