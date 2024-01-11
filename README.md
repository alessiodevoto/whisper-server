### Dockerized Whisper Automatic Speech Recognition

Deploy a speech to text demo with OpenAI Whisper Automatic Speech Recognition via Docker. 

#### Setup (in 5 minutes)

First of all, make sure you are in a directoy containing both this repository and model weights. Then, build the Docker image: 

```
docker build -t repo/whisper-server:0.0.1 -f whisper-server/Dockerfile .
```

> When building the image we copy app files and model weights from local file system. In case the directories
containing app files and models are at a different location, you should edit the dockerfile accordingly.

You can then run a container using the `run_service.sh` script. This will create a new container and start up a server inside it.

```
bash run_service.sh --port <your_port> --gpu_index <cuda_device_index> --model_size <model_size> --image_name <image_name>
```

Beware that:
- if you provide a port which is already allocated, the container will stop immediately
- `<image_name>` is the name you gave to your image when your built it.
- `<cuda_device_index>` is the index of the gpu you want to use for inferece. Defaults to 0.
- `<model_size>` is the size of the whisper model you want to use. You can choose between `tiny`, `medium` and `large-v3` (recommended) 


