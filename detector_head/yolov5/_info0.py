# -*- coding: utf-8 -*- #
# ----------------------------------------
# File Name: _info0.py
# Author: 黄俊炳
# modifier: 黄俊炳
# Version: v00
# Created: ...
# Modification: 2023/05/31
# Description: 一个小案例，用来改已经训练好的模型的一些内容
# ----------------------------------------
import torch

ckpt = torch.load(r"E:\_HK9007\yolov5-7.0\runs\hat_20230614\exp\weights\epoch10.pt")
# print(ckpt["illegal"])
ckpt["illegal"] = ["未佩戴安全帽"]
ckpt["model"].names = {0: "佩戴安全帽", 1: "佩戴安全帽", 2: "未佩戴安全帽"}
torch.save(ckpt,"epoch10.pt")

ckpt = torch.load(r"E:\_HK9007\yolov5-7.0\runs\hat_20230614\exp\weights\epoch30.pt")
# print(ckpt["illegal"])
ckpt["illegal"] = ["未佩戴安全帽"]
ckpt["model"].names = {0: "佩戴安全帽", 1: "佩戴安全帽", 2: "未佩戴安全帽"}
torch.save(ckpt,"epoch30.pt")
