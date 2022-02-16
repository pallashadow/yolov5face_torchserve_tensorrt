# Serve TensorRT Model on TorchServe
model：yolov5s  
inference library：TensorRT  
inference serve：Torchserve  

11ms latency，query per second (QPS) 700 on T4 GPU server。

# docker install and run (recommended)
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
### test
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

# Manual install and run (Without Docker)

### install torchserve and download model
see readme_torchserve.md##install

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


