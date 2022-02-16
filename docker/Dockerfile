FROM nvcr.io/nvidia/tensorrt:20.12-py3

ARG VERSION_TORCH2TRT=0.3.0

WORKDIR /app
COPY requirements.txt .

RUN python3 -m pip install pip --upgrade && \
    python3 -m pip install -r requirements.txt && \
    cd /tmp && \
    wget https://github.com/NVIDIA-AI-IOT/torch2trt/archive/refs/heads/v${VERSION_TORCH2TRT}.zip -O torch2trt-${VERSION_TORCH2TRT}.zip && \
    unzip torch2trt-${VERSION_TORCH2TRT}.zip && \
    cd torch2trt-${VERSION_TORCH2TRT} && \
    python3 setup.py install && \
    cd /tmp && rm -rf * && \
    apt-get update && \
    apt-get -y install \
        openjdk-11-jdk-headless \
        netcat

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

EXPOSE 8080 8081 8082 7070 7071

CMD [ "/app/torchserve/run.sh" ]
