Follow below instructions to deploy yolov5face

1. git clone https://gitlab.gtvorg.tk/pannixilin/yolov5face.git
2. cd yolov5face/docker
3. docker-compose up -d


Configurations
The yolov5face configurations are actually configures to torchserve. The configuration file locates at:
yolov5face/config/config.properties

The configuration items are the ip addresses and port that the service is binded to.

The worker number of torchserve is currently hard fixed to 4.


Bottlenecks:
Each yolov5face torchserve consumes 2.5G memory in average, so memory of the system is a bottleneck.