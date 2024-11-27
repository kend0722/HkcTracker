# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/24 下午5:01
@Author  : Kend
@FileName: main_demo.py.py
@Software: PyCharm
@modifier:
"""
import argparse
import os.path as osp
import cv2
import time
import os
import torch
from loguru import logger

from test.demo.demo import image_demo, imageflow_demo
from detector_head.predictor import Predictor
from track_utils.model_utils import get_model_info, fuse_model
from track_utils.my_timer import MyTimer
from visualization.build import get_exp
from visualization.visualize import plot_tracking


# 解析命令行参数
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default=r"D:\kend\WorkProject\Hk_Tracker\data\dataset\test_images", help="path to "
                                                                                                            "images or video")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of "
                                                                   "image/video",)
    # exp file
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file",)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device",default="cpu",type=str,help="device to run our model, can either be cpu or gpu",)
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=5, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16",dest="fp16",default=False,action="store_true", help="Adopting mix precision "
                                                                                     "evaluating.",)
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt",dest="trt",default=False,action="store_true", help="Using TensorRT model for testing.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of "
                                                                               "which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
