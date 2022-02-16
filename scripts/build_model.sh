#!/bin/bash

SCRIPT_DIR="$(dirname $(readlink -f ${0}))"
ROOT_DIR="$(readlink -f ${SCRIPT_DIR}/..)"
OUT_DIR="${ROOT_DIR}/weights"

# DOCKER_BUILDKIT=1 cause build failure
# because BuildKit does allow to access gpu.
# see issue https://github.com/moby/buildkit/issues/1800
# DOCKER_BUILDKIT=1 docker build --target export_model -f docker/Dockerfile -o "${OUT_DIR}" .
# use standard docker build instead
docker build -t yolov5face:model -f ${ROOT_DIR}/docker/Dockerfile_model ${ROOT_DIR} && \
docker create -it --name yolov5face_model_tmp yolov5face:model bash && \
docker cp yolov5face_model_tmp:/build/weights/yolov5s-face.torch2trt "${OUT_DIR}" && \
docker rm yolov5face_model_tmp && \
docker rmi yolov5face:model
