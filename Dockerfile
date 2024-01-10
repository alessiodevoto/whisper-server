FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && \
    apt install -y bash \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/
WORKDIR /workspace/
# The paths from where we copy the model and the app are relative 
# to the directory where the 
# docker build command is executed.
COPY ./speech-to-text-app/fast-whisper/ ./fast-whisper
COPY ./service_workspace/models/ ./models


RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/fast-whisper/fast_whisper_reqs.txt 
