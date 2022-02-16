
# Use Torchserve and TensorRT (torch2trt) and yolov5face to build a face detection server

use Torchserve with TensorRT backend（recommended） with 11ms latency，query per second (QPS) 700 on T4 GPU server

use Torchserve with Jit TorchScript backend see **torchserve/readme_torchserve_depricated.md**. With higher latency and lower throughput. 

this repo is adapted from https://github.com/deepcam-cn/yolov5-face (Warning: GNU LICENSE)
1. Add Torchserve as Inference server
2. accerlerated with TensorRT by torch2trt toolkit, with 10x lower latency and 2x larger throughput. this is the first demo to show how serve TensorRT model on Torchserve as far as I know. 
3. add Docker and logging. 

Where Torchserve is a performant, flexible and easy to use tool for serving PyTorch eager mode and torschripted models. TensorRT is a library developed by NVIDIA for faster inference on NVIDIA graphics processing units (GPUs). ... It can give around 4 to 5 times faster inference on many real-time services and embedded applications. torch2trt is a PyTorch to TensorRT converter which utilizes the TensorRT Python API. It remain the input/ouput of the model as Torch Tensor format. https://github.com/NVIDIA-AI-IOT/torch2trt


## why choose yolov5face as face detection model
1. currently (20210904) a SOTA face-detection model on widerface benchmark, balanced between speed and accuracy. 
2. based on pytorch，easy to finetuning，easy to build inference server via torchserve. 

## model local test and visualization 
`
python torchserve/qpstest.py --mode 1 --vis 1 --image data/images/test.jpg
`

## Torhcserve pipeline
1、decode the image from jpg，resize and padding to lower resolution as 320×320 for acceleration。
2、batch inference with TensorRT backend
3、revert face coords to the size of original resolution and return result with json format

## Interface protocal

input：jpg binary data，
output：json format, (bounding box, confidence, 5 landmarks)
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
every image consists of N faces，every face include 3 keys: 
- xywh_ratio is face center coordinates and width and height, as ratio of the image size. 
- conf is the confidence of face detection from 0 to 1  
- landmarks_ratio are 5 coords of face landmarks, as ratio of the image size


# QuickStart with Docker

Follow below instructions to deploy yolov5face

1. cd yolov5face/docker
2. docker-compose up -d

Configurations
The yolov5face configurations are actually configures to torchserve. The configuration file locates at:
yolov5face/config/config.properties

The configuration items are the ip addresses and port that the service is binded to.

The worker number of torchserve is currently hard fixed to 4.


Bottlenecks:
Each yolov5face torchserve consumes 2.5G memory in average, so memory of the system is a bottleneck.


# Install Manually without Docker

### install dependencies
```
pip install -r requirements
```
install java11 dependence.    https://www.ubuntu18.com/ubuntu-install-openjdk-11/  
On cloud server, if cuda version is different from cuda10.2, manually edit the pytorch version in requirements.txt
https://pytorch.org/get-started/locally/

### Download model file 
download 50M file **yolov5s**   https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing  
unzip to folder 
```
weights/yolov5s-face.pt
```

### install TensorRT without docker
download TensorRT-7 (compatible with torch2trt tool on 2021, maybe TensorRT-8 is also compatible for now)
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
recommend to install via tar.gz, which is compatible with conda environment
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
be aware to write correct ubuntu version，cuda version，and cudnn version

### install torch2trt
```
cd ~/
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

## Start Server and Register Model
1. pack models and python code to torchserve .mar format. Backends with TensorRT. 
```
python ./torchserve/model_repack.py --trt 1
```
will generate a file "./torchserve/model_store/trt_fd1.mar".   
- start server
```
torchserve --start --ncs --model-store ./torchserve/model_store/
```
- localhost register model 
```
curl -X POST "127.0.0.1:8081/models?url=trt_fd1.mar&batch_size=1&max_batch_delay=2&initial_workers=4&model_name=fd1"
```
Note that  
1) url=trt_fd1.mar  
2) batch_size=1  
3) initial_workers=2
where 2 is the number of cpu cores on your server, and require 3 * 2 GB system memory.   



# docker install and run
### git clone torch2trt
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt --branch v0.3.0
```
### pull nvidia tensorrt7 docker image, (包含ubuntu20.04, cuda, cudnn, tensorrt7.2.2), 时间较长
```
docker pull nvcr.io/nvidia/tensorrt:20.12-py3
```
### install nvidia-container-runtime
why this step？ docker build need GPU and torch2trt to convert the model，see https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime  
1. Install nvidia-container-runtime:
```
sudo apt-get install nvidia-container-runtime
```
2. Edit/create the **/etc/docker/daemon.json** with content:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
3. Restart docker daemon:
```
sudo systemctl restart docker
```

### build environment docker image
```
docker build -f Dockerfile_base --tag base --progress=plain .
```
### build final docker image
```
docker build -f Dockerfile_torchserve_tensorrt --tag ts_trt --progress=plain .
```
### run
```
docker run --gpus all -it --rm --name test -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 ts_trt
```
It will run torchserve/run.sh inside the image. 
1、start torchserve
2、register the model
success log:   
{"status": "Model \"fd1\" Version: 1.0 registered with 4 initial workers"}  


# TEST
QPS test
```
python torchserve/qpstest.py --mode 3
```
visualization
```
python torchserve/qpstest.py --mode 3 --vis 1
```
### API example
`
python torchserve/api.py
`
or  
`
curl 127.0.0.1:8080/predictions/fd1 -T ./data/images/zidane.jpg  
`
### QPS test on torchserve
```
bash ./torchserve/run.sh
python torchserve/qpstest.py --mode 3
```
QPS test on local torchscript model
```
python ./torchserve/model_repack.py --trt 0
python torchserve/qpstest.py --mode 1
```
QPS test on local tensorrt model
```
python ./torchserve/model_repack.py --trt 1
python torchserve/qpstest.py --mode 2
```


