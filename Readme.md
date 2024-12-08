## 这是一个基于ByteTrack和YOLO的多目标跟踪项目

## 项目介绍
本项目是一个基于ByteTrack和YOLO的多目标跟踪项目，它使用YOLOv5或者Yolo11作为目标检测头，使用ByteTrack作为多目标跟踪算法。为了方便理解我的项目，我把检测头和跟踪算法拆分开了
总所周知YOLO是一个流行的目标检测模型，而ByteTrack是一个高效的多目标跟踪算法。在使用cpu模式下，ByteTrack的FPS可以达到15帧左右，而YOLOv5的FPS可以达到20帧左右。

## 项目结构
本项目包含以下文件和目录：
- `data/`：存放测试数据集。
- -`detector_head/`：目标检测头，支持的有yolov5和yolo11，当然你可以在里面按照你的需求更改
- `predictor.py/`：你的具体目标检测头实现的基类。
- `fastapi`：fastapi，我们考虑采用海葵云的分布式平台去部署，所以留了一个实现api的目录
- `test`：包涵了一些demo和功能测试，可以只用看demo，因为test目录是我个人的习惯
- `tracking`：ByteTrack算法的实现模块，用于多目标跟踪。
- `byte_tracker.py`：ByteTrack算法的主要实现类，需要多次阅读代码
- `utils`：一些其他的工具包和组件
- `main.py`：主程序。
- `Readme.md`：项目说明文档。
- `requirements.txt`：项目依赖的Python库。

## 注意
这只是一个示例demo，实际使用中可能需要根据具体需求进行修改。
项目的健壮性和性能还受到很多因素的影响，例如检测头、跟踪算法、数据集等。

## 致谢
本项目基于以下项目进行了修改和优化：
- [YOLOv5](https://github.com/ultralytics/yolov5)：yolov5目标检测模型。
- [ByteTrack](https://github.com/ifzhang/ByteTrack)：ByteTrack多目标跟踪算法。
- [ultralytics/yol011](https://github.com/ultralytics/ultralytics)：YOLO11的官方实现。
- [HaiKuicloud](https://lyh.haikuicloud.com): 特别需要感谢海葵云部门，为我们搭建了分布式集群，能够很好的部署我们的算法

## 记录的问题点
1、ByteTrack的update方法接收n*7的张量数据, 包涵xywh+对象置信度+类别置信度+具体的类别，而yolov5的predict方法返回的n*6的列表，包涵xywh和置信度
所以我改了下NMS，将置信度拆成了对象置信度+类别置信度，去对齐跟踪器，并且没有进行缩放回原图尺寸。(因为跟踪器回帮我们做这一部分内容)
在实际demo中我没有使用缩放并且返回了n*7，而是直接对齐了，所以可能存在一些问题，但是基本可以正常跟踪了。
2、YOLOv5的检测头可能在复杂背景（如遮挡、多目标密集区域）或小目标检测方面性能不足，导致目标容易被误检或漏检。
YOLOv5 的非极大值抑制（NMS）可能导致目标检测框不够稳定，影响多目标跟踪的准确性。
3、ByteTrack算法非常依赖高质量的检测头，所以检测头做好了跟踪的效果自然就好。所以我决定替换为yolo11进行测试。由于项目的人体检测模型是基于yolov5的
所以我直接在yolo11上进行了修改，使其可以输出检测框，并且输出的检测框是经过nms处理后的，并且进行了归一化。
4、ByteTrack算法的FPS在cpu模式下可以达到15帧左右，而YOLOv5的FPS可以达到20帧左右，实际项目需要根据具体需求进行优化。
5、完整的项目逻辑我还在开发中，目前只实现了检测头和跟踪算法的调用，没有实现完整的项目逻辑，后续会继续完善，敬请期待。


