# Serve Yolov5face Pytorch Model on TorchServe
模型：yolov5s  
神经网络inference框架：TorchScript  
服务框架：Torchserve  

## install (ubuntu)
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

## Quickstart
### Start Server and Register Model
1. pack models and python code to torchserve .mar format
```
python ./torchserve/model_repack.py
```
will generate a file "./torchserve/model_store/jit_fd1.mar".   
- start server
```
torchserve --start --ncs --model-store ./torchserve/model_store/
```
- localhost register model 
```
curl -X POST "127.0.0.1:8081/models?url=jit_fd1.mar&batch_size=32&max_batch_delay=2&initial_workers=2&model_name=fd1"
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
python torchserve/qpstest.py --mode 2
```
QPS test on local model
```
python torchserve/qpstest.py --mode 1
```
## other torchserve commands
unregister model
```
curl -X DELETE http://localhost:8081/models/fd1
```
check model status
```
curl 127.0.0.1:8081/models
curl 127.0.0.1:8081/models/fd1
```
stop server
```
torchserve --stop
```
## log dir
`
./logs/
`

## Reference
torchserve documentations  
https://github.com/pytorch/serve  
torchserve + docker  
https://github.com/pytorch/serve#quick-start-with-docker  
