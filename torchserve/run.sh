#!/bin/bash

# python3 ./torchserve/model_repack.py --trt 1 # 这一步通过nvidia-container-runtime 放在docker build中
SCRIPT_DIR="$(dirname $(readlink -f ${0}))"
ROOT_DIR="$(readlink -f ${SCRIPT_DIR}/..)"


torchserve --start --ncs --ts-config ${ROOT_DIR}/config/config.properties --model-store ${ROOT_DIR}/torchserve/model_store/ # start torchserve, log_dir=./logs

echo "********************************************************"
echo "Waiting for the torchserve to start on port 8080 and 7070"
echo "********************************************************"
while ! `nc -vz localhost 7070`; do sleep 3; done
echo "******* torchserve has started"

curl -X POST "127.0.0.1:8081/models?url=trt_fd1.mar&batch_size=1&max_batch_delay=2&initial_workers=4&model_name=fd1" # register model

tail -f /dev/null # 阻止自动退出
