
# yolov5face人脸检测，和torchserve服务搭建

使用TensorRT的安装和启动方法见**readme_torchserve_tensorrt.md** （推荐）  

使用TorchScript的安装和启动方法见
**torchserve/readme_torchserve_depricated.md** （速度相对较慢）

此项目 adapted from https://github.com/deepcam-cn/yolov5-face  
在原项目上增加了torchserve服务，包含基于torchserve的工程优化和日志系统。  

## 模型为什么选择yolov5face
1. 是目前(20210904) widerface benchmark上的综合得分最高的项目（综合速度和精度）。
2. 基于pytorch，方便修改和重新训练，方便基于torchserve快速搭建服务。
3. 模型尺寸选用gpu系列计算量最小的yolov5s，实验表明精度满足要求  

本地运行，可视化  
`
python torchserve/qpstest.py --mode 1 --vis 1 --image data/images/test.jpg
`

## 基本原理
1、服务器多进程将图片jpg解码，resize和padding到320×320尺寸。
2、服务器批量预测人脸位置
3、服务器多进程将人脸位置还原到原图尺寸，返回json结果

## 服务器调用 example
`
curl localhost:8080/predictions/fd1 -T ./data/images/zidane.jpg
`

python接口及可视化  
`
python torchserve/api.py
`

## 接口规范

输入：jpg数据流，  
支持两种通信方式  
http post 默认端口8080  
grpc默认端口7070  
输出：例如
```json
[
	{
		"xywh_ratio": [0.7689772367477417, 0.25734335581461587, 0.11677041053771975, 0.26296865675184466], 
	    "conf": 0.8641895651817322, 
    	"landmarks_ratio": [0.754405927658081, 0.22680193583170574, 0.8030961990356446, 0.23478228251139324, 0.7799828529357911, 0.2754765404595269, 0.7510656356811524, 0.31618389553493925, 0.7911150932312012, 0.32295591566297743]
    }, 
    {
    	"xywh_ratio": [0.4645264148712158, 0.47456512451171873, 0.12120456695556636, 0.29619462754991316], 
        "conf": 0.7263935804367065, 
        "landmarks_ratio": [0.4809267997741699, 0.44996253119574653, 0.5082815647125244, 0.4542162577311198, 0.5047649383544922, 0.5095860799153645, 0.4696146011352539, 0.5512683444552952, 0.4905359745025635, 0.5559690687391493]
    }
]
```
包含N张人脸，其中每张人脸包含3个字段：  
- xywh_ratio为人脸中心点位置和宽高4个数，单位为[相对图片比例]  
- conf为置信读,0-1  
- landmarks_ratio为5关键点坐标，单位为[相对图片比例]  
## TODO
1、本地速度测试，和python QPS测试脚本（完成）  
2、torchscript加速（完成）  
3、搭建torchserve服务(完成)，启动方法见readme_torchserve.md  
4、本地测试tensorRT版本，在批量状态下的QPS， (完成)  
5、在服务器（T4显卡）测试tensorRT版本QPS，评估是否需要改用TensorRT部署。  （完成）结论是tensorRT无论QPS和throughput都比torchscript有显著优势。  
6、搭建以TensorRT模型为backend的torchserve服务（完成）启动方法见readme_torchserve_tenserrt.md

## torchserve简介
Torchserve是facebook官方支持的基于pytorch的inference服务解决方案。优点是开发简单，改动方便。
## TensorRT简介
TensorRT是基于Nvidia官方支持的通用inference服务解决方案，单张图片的推理速度能达到极限。适合对latency要求高的实时系统。
而pytorch或torchscript的缺点是图片数量少时，单张延迟较高。批量处理虽然throughput也比较大，但单张图片latency较长，适合服务器批量处理图片。
## torch2trt简介
torch2trt是nvidia官方提供的一个将pytorch模型转变成tensorRT模型，但不改变接口的工具。
https://github.com/NVIDIA-AI-IOT/torch2trt


以下是原项目readme 20210904  ，原项目是论文对应的训练框架，不包括服务。警告：原项目是GNU LICENSE  
----------------------------------------
## What's New

**2021.08**: Yolov5-face to TensorRT.  
Inference time on rxt2080ti.
|Backbone|Pytorch |TensorRT_FP16 |
|:---:|:----:|:----:|
|yolov5n-0.5|11.9ms|2.9ms|
|yolov5n-face|20.7ms|2.5ms|
|yolov5s-face|25.2ms|3.0ms|
|yolov5m-face|61.2ms|3.0ms|
|yolov5l-face|109.6ms|3.6ms|
> Note: (1) Model inference  (2) Resolution 640x640


**2021.08**: Add new training dataset [Multi-Task-Facial](https://drive.google.com/file/d/1Pwd6ga06cDjeOX20RSC1KWiT888Q9IpM/view?usp=sharing),improve large face detection.
| Method               | Easy  | Medium | Hard  | 
| -------------------- | ----- | ------ | ----- |
| ***YOLOv5s***        | 94.56 | 92.92  | 83.84 |
| ***YOLOv5m***        | 95.46 | 93.87  | 85.54 |


## Introduction

Yolov5-face is a real-time,high accuracy face detection.

![](data/images/yolov5-face-p6.png)

## Performance

Single Scale Inference on VGA resolution（max side is equal to 640 and scale).

***Large family***

| Method              | Backbone       | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
| :------------------ | -------------- | ----- | ------ | ----- | ----------- | ---------- |
| DSFD (CVPR19)       | ResNet152      | 94.29 | 91.47  | 71.39 | 120.06      | 259.55     |
| RetinaFace (CVPR20) | ResNet50       | 94.92 | 91.90  | 64.17 | 29.50       | 37.59      |
| HAMBox (CVPR20)     | ResNet50       | 95.27 | 93.76  | 76.75 | 30.24       | 43.28      |
| TinaFace (Arxiv20)  | ResNet50       | 95.61 | 94.25  | 81.43 | 37.98       | 172.95     |
| SCRFD-34GF(Arxiv21) | Bottleneck Res | 96.06 | 94.92  | 85.29 | 9.80        | 34.13      |
| SCRFD-10GF(Arxiv21) | Basic Res      | 95.16 | 93.87  | 83.05 | 3.86        | 9.98       |
| -                   | -              | -     | -      | -     | -           | -          |
| ***YOLOv5s***       | CSPNet         | 94.67 | 92.75  | 83.03 | 7.075       | 5.751      |
| **YOLOv5s6**        | CSPNet         | 95.48 | 93.66  | 82.8  | 12.386      | 6.280      |
| ***YOLOv5m***       | CSPNet         | 95.30 | 93.76  | 85.28 | 21.063      | 18.146     |
| **YOLOv5m6**        | CSPNet         | 95.66 | 94.1   | 85.2  | 35.485      | 19.773     |
| ***YOLOv5l***       | CSPNet         | 95.78 | 94.30  | 86.13 | 46.627      | 41.607     |
| ***YOLOv5l6***      | CSPNet         | 96.38 | 94.90  | 85.88 | 76.674      | 45.279     |


***Small family***

| Method               | Backbone        | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) |
| -------------------- | --------------- | ----- | ------ | ----- | ----------- | ---------- |
| RetinaFace (CVPR20   | MobileNet0.25   | 87.78 | 81.16  | 47.32 | 0.44        | 0.802      |
| FaceBoxes (IJCB17)   |                 | 76.17 | 57.17  | 24.18 | 1.01        | 0.275      |
| SCRFD-0.5GF(Arxiv21) | Depth-wise Conv | 90.57 | 88.12  | 68.51 | 0.57        | 0.508      |
| SCRFD-2.5GF(Arxiv21) | Basic Res       | 93.78 | 92.16  | 77.87 | 0.67        | 2.53       |
| -                    | -               | -     | -      | -     | -           | -          |
| ***YOLOv5n***        | ShuffleNetv2    | 93.74 | 91.54  | 80.32 | 1.726       | 2.111      |
| ***YOLOv5n-0.5***    | ShuffleNetv2    | 90.76 | 88.12  | 73.82 | 0.447       | 0.571      |



## Pretrained-Models

| Name        | Easy  | Medium | Hard  | FLOPs(G) | Params(M) | Link                                                         |
| ----------- | ----- | ------ | ----- | -------- | --------- | ------------------------------------------------------------ |
| yolov5n-0.5 | 90.76 | 88.12  | 73.82 | 0.571    | 0.447     | Link: https://pan.baidu.com/s/1UgiKwzFq5NXI2y-Zui1kiA  pwd: s5ow, https://drive.google.com/file/d/1XJ8w55Y9Po7Y5WP4X1Kg1a77ok2tL_KY/view?usp=sharing |
| yolov5n     | 93.61 | 91.52  | 80.53 | 2.111    | 1.726     | Link: https://pan.baidu.com/s/1xsYns6cyB84aPDgXB7sNDQ  pwd: lw9j,https://drive.google.com/file/d/18oenL6tjFkdR1f5IgpYeQfDFqU4w3jEr/view?usp=sharing |
| yolov5s     | 94.33 | 92.61  | 83.15 | 5.751    | 7.075     | Link: https://pan.baidu.com/s/1fyzLxZYx7Ja1_PCIWRhxbw  Link: eq0q,https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing |
| yolov5m     | 95.30 | 93.76  | 85.28 | 18.146   | 21.063    | Link: https://pan.baidu.com/s/1oePvd2K6R4-gT0g7EERmdQ  pwd: jmtk |
| yolov5l     | 95.78 | 94.30  | 86.13 | 41.607   | 46.627    | Link: https://pan.baidu.com/s/11l4qSEgA2-c7e8lpRt8iFw  pwd: 0mq7 |

## Data preparation

1. Download WIDERFace datasets.
2. Download annotation files from [google drive](https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view?usp=sharing).

```shell
python3 train2yolo.py
python3 val2yolo.py
```



## Training

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" python3 train.py --data data/widerface.yaml --cfg models/yolov5s.yaml --weights 'pretrained models'
```



## WIDERFace Evaluation

```shell
python3 test_widerface.py --weights 'your test model' --img-size 640

cd widerface_evaluate
python3 evaluation.py
```

#### Test

![](data/images/result.jpg)


#### Android demo

https://github.com/FeiGeChuanShu/ncnn_Android_face/tree/main/ncnn-android-yolov5_face

#### opencv dnn demo

https://github.com/hpc203/yolov5-dnn-cpp-python-v2


#### References

https://github.com/ultralytics/yolov5

https://github.com/DayBreak-u/yolo-face-with-landmark

https://github.com/xialuxi/yolov5_face_landmark

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/deepinsight/insightface


#### Citation 
- If you think this work is useful for you, please cite 
 
      @article{YOLO5Face,
      title = {YOLO5Face: Why Reinventing a Face Detector},
      author = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
      booktitle = {ArXiv preprint ArXiv:2105.12931},
      year = {2021}
      }

#### Main Contributors
https://github.com/derronqi  

https://github.com/changhy666 

https://github.com/bobo0810 

