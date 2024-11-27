# -*- coding: utf-8 -*- #
# ----------------------------------------
# File Name: demo_seg.py
# Author: 黄俊炳
# modifier: 黄俊炳
# Version: v00
# Created: ...
# Modification: 2023/05/31
# Description: 一个小案例，用来演示yolov5的用法
# ----------------------------------------
from yolov_func import *
from yolov_model import *


def iou(location, other, illegal):
    bboxes = np.array(other)[:, 2:].astype(np.int16)
    x1 = bboxes[:, 0]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    xx1 = np.maximum(int(location[2]), x1)
    yy1 = np.maximum(int(location[3]), y1)
    xx2 = np.minimum(int(location[4]), x2)
    yy2 = np.minimum(int(location[5]), y2)
    w = np.maximum(0.0, (xx2 - xx1))
    h = np.maximum(0.0, (yy2 - yy1))
    inter = w * h
    iou = inter / areas
    ids = np.where(iou >= 0.9)[0]
    output_illegal = []
    output_legal = []
    for i in ids:
        re = [other[i][0], other[i][1], (other[i][2], other[i][3]), (other[i][4], other[i][5])]
        if other[i][0] in illegal:
            output_illegal.append(re)
        else:
            output_legal.append(re)
    return output_illegal, output_legal


if __name__ == "__main__":
    # weights = r"E:\_HK9007\__hkframe\distributed_v00\server\weights\sseg_v01.pt"
    # weights = r"E:\_HK9007\__hkframe\distributed_v00\server\weights\rope_ladder_s_v01.pt"
    weights = r"E:\_HK9007\__hkframe\distributed_v00\server\weights\xseg_v03.pt"
    # 服务器数据
    # device = torch.device('cpu') # 0:05:47.132524
    device = torch.device('cuda:0')  # 0:00:17.514154s 500次  # 0:00:13.347790 极速0:00:13.347790
    model = DetectMultiBackend(weights, device=device)

    # #
    # stride, names, pt = model.stride, model.names, True
    # imgsz = [640, 640]
    # print(stride, names)
    # import os
    #
    # for path in os.listdir("img"):
    #     im0 = cv2.imread(r"img\{}".format(path))  # BGR
    #     im = letterbox(im0, imgsz, stride, auto=pt)[0]  # padded resize
    #     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     im = np.ascontiguousarray(im)  # contiguous
    #     im = torch.from_numpy(im).to(model.device)
    #     im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    #     im /= 255  # 0 - 255 to 0.0 - 1.0
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    #
    #     pred, proto = model(im)[:2]
    #     conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.7, 0.45, None, False, 20
    #     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=0)
    #     output_ladder = []
    #     output_other = []
    #     for i, det in enumerate(pred):
    #         if len(det):
    #             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
    #             for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
    #                 x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    #                 p1, p2 = (x1, y1), (x2, y2)
    #                 cls = names[int(cls)]
    #                 if cls == "ladder":
    #                     y1 = y1 - (y2 - y1) * 0.5
    #                     y1 = int(y1) if y1 > 0 else 0
    #                     output_ladder.append([cls, round(float(conf), 4), x1, y1, x2, y2])
    #                     cv2.rectangle(im0, p1, p2, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    #                 elif cls == "rope":
    #                     cv2.rectangle(im0, p1, p2, (0, 125, 0), thickness=3, lineType=cv2.LINE_AA)
    #                     output_other.append([cls, round(float(conf), 4), x1, y1, x2, y2])
    #                 else:
    #                     cv2.rectangle(im0, p1, p2, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    #                     output_other.append([cls, round(float(conf), 4), x1, y1, x2, y2])
    #     results = {"illegal": [], "legal": []}
    #     if len(output_other) != 0:
    #         for ladder in output_ladder:
    #             results_illegal, results_legal = iou(ladder, output_other, ['rope'])
    #             results["illegal"] += results_illegal
    #             results["legal"] += results_legal
    #     cv2.imwrite(r"output\{}".format(path), im0)
