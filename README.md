### Whisper Server

Deploy a demo OpenAI Whisper Automatic Speech Recognition via Docker. 


First of all, make sure you are in a directoy containing this repository and model weights. Then, build the Docker image with 

```
docker build -t repo/whisper-server:0.0.1 -f whisper-server/Dockerfile .
```

Notice that when building the image we copy app files and model weights from local file system. In case the directories
containing app files and models are at a different location, you should edit the dockerfile accordingly.

You can then run a container like so:

```
sudo docker run --name whisper-service -p 4022:4022 --gpus all  -dit repo/whisper-server:0.0.1 
```

Finally, you can run the the command to start the server from inside the container.

