# Serve TensorRT Model on TorchServe
模型：yolov5s  
神经网络inference框架：TensorRT  
服务框架：Torchserve  

用torchserve挂载TensorRT模型，throughput更高，可有2倍提升。latency更低，可有10倍提升。  
本地测试latency平均11ms，预计16cpu1gpu QPS大约在700左右。

# docker install and run (recommended)
### git clone torch2trt项目到根目录
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt --branch v0.3.0
```
### 拉取nvidia tensorrt7 docker image, (包含ubuntu20.04, cuda, cudnn, tensorrt7.2.2), 时间较长
```
docker pull nvcr.io/nvidia/tensorrt:20.12-py3
```
### 安装nvidia-container-runtime
为什么有这一步？因为docker build需要通过torch2trt对模型转换，build过程中需要依赖GPU，参考  
https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime）  
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

### 在image中安装依赖，生成环境image
```
docker build -f Dockerfile_base --tag base --progress=plain .
```
### 生成最终image
```
docker build -f Dockerfile_torchserve_tensorrt --tag ts_trt --progress=plain .
```
### 运行image
```
docker run --gpus all -it --rm --name test -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 ts_trt
```
会在docker内部运行torchserve/run.sh脚本
1、启动torchserve  
2、挂载模型。  
出现以下信息说明模型挂载成功  
{"status": "Model \"fd1\" Version: 1.0 registered with 4 initial workers"}  
### 测试
QPS测试
```
python torchserve/qpstest.py --mode 3
```
可视化
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

# Manual install and run (Without Docker)
以下是不通过docker启动的手动安装和启动方式
## install (ubuntu)
### install torchserve and download model
see readme_torchserve.md##install

### install TensorRT
从官方网站注册，并下载最新版TensorRT-7 (TensorRT-8好像不太稳定，有bug)
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
由于可能会在conda环境启动，建议采用tar.gz的安装方式
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
警告：安装时需要密切注意服务器的ubuntu版本，cuda版本，和cudnn版本不能填错。

### install torch2trt
注意：这里采用了额外的NVIDIA-AI-IOT团队的转换工具，操作非常简单。  
没有采用原项目中./torch2tensorrt文件夹下的传统转换方案。
```
cd ~/
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

## 启动服务
### Start Server and Register Model
1. pack models and python code to torchserve .mar format. Backends with TensorRT. 改用torch2trt工具生成模型并打包。
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
3) initial_workers=N  
where N is the number of cpu cores on your server, and require 3 * N GB system memory.   
默认服务器1张显卡  
这个参数N是GPU进程数（在显卡上启动模型的数量）视服务器规格而定，一般N<=服务器cpu数量，会占用3×N GB内存。在CPU cores和memory足够的情况下越大越好

