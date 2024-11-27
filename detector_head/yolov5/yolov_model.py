# -*- coding: utf-8 -*- #
# ----------------------------------------
# File Name: yolov5_model.py
# Author: 黄俊炳
# modifier: 黄俊炳
# Version: v00
# Created: ...
# Modification: 2023/05/31
# Description: yolov5框架所用到的类，勿随意改动，详细作用请自行了解yolov5框架
# ----------------------------------------
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class DetectMultiBackend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, fp16=False, data=None, fuse=True):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        fp16 &= True
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        #
        model = Ensemble()
        file = Path(str(weights).strip().replace("'", ''))
        ckpt = torch.load(file)
        names = ckpt["model"].names
        illegal = ckpt.get("illegal", [])  # 新增违规类型
        ckpt = (ckpt.get('ema') or ckpt['model']).to("cpu").float()  # FP32 model
        ckpt.stride = torch.tensor([32.])
        model.append(ckpt.fuse().eval())  # model in eval mode
        for m in model.modules():
            m.inplace = True
        model = model[-1]
        stride = max(int(model.stride.max()), 32)  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        model.half() if fp16 else model.float()
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear},
        #                                             dtype=torch.qint8)

        print(f"detect model to {device}")
        self.model = model.to(device)  # explicitly assign for to(), cpu(), cuda(), half()
        #
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        y = self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
