#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2023/12/03
@Time: 15:19
@Description: post_process - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""
import torch
import torchvision


"""
如果一个预测框的置信度是0.9，表示模型认为这个框内很可能包含一个对象。然后，模型预测这个对象属于 "猫" 类别的概率是0.8，属于 "狗" 类别的概率是0.2。
那么，这个预测框内对象属于 "猫" 类别的类别置信度就是 0.9 * 0.8 = 0.72，属于 "狗" 类别的类别置信度就是 0.9 * 0.2 = 0.18。
总的来说，置信度是对预测框内是否存在对象的总体评价，而类别置信度则是对预测框内对象属于某个具体类别的评价。
"""


"""图像后处理, 支持图像的批处理，取决于prediction的batch_size"""
def postprocess(prediction, num_classes, conf_thre=0.1, nms_thre=0.3):
    """
    Args:
        prediction: 模型的预测输出，形状为 (batch_size, num_boxes, 5 + num_classes)。其中，前4个值表示边界框的中心点坐标和宽度高度，第5个值表示对象置信度，剩下的值表示各个类别的概率。
        num_classes: 类别数量。
        conf_thre: 置信度阈值，用于过滤掉低置信度的预测框，默认值为 0.5。
        nms_thre: 非极大值抑制的阈值，默认值为 0.45。
    Returns:
        # example:
            output = [
                torch.tensor([
                    [10.0, 20.0, 50.0, 60.0, 0.9, 0.8, 1.0],  # 第一个图像的第一个检测框 + 置信度+ 类别置信度+ 类别索引
                    [100.0, 150.0, 200.0, 250.0, 0.85, 0.75, 2.0]  # 第一个图像的第二个检测框
                ]),
                torch.tensor([
                    [15.0, 25.0, 55.0, 65.0, 0.88, 0.82, 3.0],  # 第二个图像的第一个检测框
                    [110.0, 160.0, 210.0, 260.0, 0.8, 0.7, 1.0]  # 第二个图像的第二个检测框
                ])
            ]
            # object_conf: 对象置信度。class_conf: 类别置信度。class_pred: 类别索引（整数）。
    """


    # print("prediction.shape", prediction.shape)  # # prediction.shape torch.Size([1, 25200, 6])  640*640+box+class+score
    box_corner = prediction.new(prediction.shape)  #xywh -> xyxy
    # # 将边界框的中心点坐标和宽度高度转换为边界框的四个角点坐标
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 左上角的 x 坐标。
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # 左上角的 y 坐标。
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # 右下角的 x 坐标。
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # 右下角的 y 坐标。
    prediction[:, :, :4] = box_corner[:, :, :4]     # 将转换后的边界框坐标赋值回 prediction 中

    # print(len(prediction))  # 1  代表一张图， 三维数组只返回最外层的列表长度，其实就是batch_size
    # 创建一个长度为 batch_size 的列表 output，用于存储每张图的最终检测结果,
    output = [None for _ in range(len(prediction))]
    # 处理每张图像的预测结果：
    for i, image_pred in enumerate(prediction):
        # print("image_pred", image_pred)  # (batch_size, num_features) 批量大小+特征数量。
        # 如果图像没有预测框，则跳过该图像。
        if not image_pred.size(0):
            continue
        # 获取最高置信度的类别的索引。和置信度：
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)  #
        # print(class_conf, class_pred)
        # 计算每个预测框的置信度与类别置信度的乘积，和置信度阈值conf_thre 进行过滤。
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()  # 布尔数组
        # 将边界框坐标、对象置信度、类别置信度和类别索引组合成一个新的张量 detections。
        # NOTE: 是这里决定了输出是七个特征值 边界框的坐标和置信度，class_conf 类别置信度，class_pred.float() 包含类别索引。因此，总共是7个特征值。
        # 而原生的yolov5是只保留了置信度
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # # 然后使用 conf_mask 过滤掉低置信度的预测框。
        detections = detections[conf_mask]
        # 如果没有剩余的预测框，则跳过当前图像。
        if not detections.size(0):
            continue
        # NMS 使用 torchvision.ops.batched_nms 进行非极大值抑制，去除重叠的预测框。
        # nms_out_index 是经过 NMS 后保留下来的预测框的索引。
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        # 将经过 NMS 后的预测框赋值给 detections。
        detections = detections[nms_out_index]
        # 将经过 NMS 处理后的检测结果存储到 output 列表中。如果 output[i] 已经有内容，则将其与新的检测结果合并。
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    # 返回每个图像的最终检测结果。NOTE: output 是一个列表，每个元素是一个张量，表示一张图像的检测结果。且没有放缩为原图的尺寸
    return output
