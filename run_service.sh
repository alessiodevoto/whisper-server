#!/bin/bash

# This script runs a Docker container with the Whisper server. The parameters of the server can be set by passing
# command-line arguments to this script. The arguments are then passed to the Python script that runs the server.
# IMPORTANT: this script assumes the image name is label/whisper-server:0.0.1

# Set default values
PORT=4020
GPU_INDEX=0
MAX_THREADS=40
LOGS_DIR="default_logs_dir"
ENABLE_CORRECTIONS=""
VERBOSE=""
LIVE=""
MODEL_SIZE="tiny"
INTERVAL=2
ONLINE=""
BATCH_SIZE=16
CHUNK_LEN=30
HF_TOKEN="hf_token"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT=$2
            shift
            ;;
        --gpu_index)
            GPU_INDEX=$2
            shift
            ;;
        --max_threads)
            MAX_THREADS=$2
            shift
            ;;
        --logs_dir)
            LOGS_DIR=$2
            shift
            ;;
        --enable_corrections)
            ENABLE_CORRECTIONS="--enable_corrections"
            ;;
        --verbose)
            VERBOSE="--verbose"
            ;;
        --live)
            LIVE="--live"
            ;;
        --model_size)
            MODEL_SIZE=$2
            shift
            ;;
        --interval)
            INTERVAL=$2
            shift
            ;;
        --online)
            ONLINE="--online"
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift
            ;;
        --chunk_len)
            CHUNK_LEN=$2
            shift
            ;;
        --hf_token)
            HF_TOKEN=$2
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Run the Python script with the parsed arguments
echo "Service will be up in roughly 30 seconds on --port $PORT --gpu_index $GPU_INDEX --model_size $MODEL_SIZE $ONLINE $ENABLE_CORRECTIONS $VERBOSE"

script_command="bash /workspace/whisper-server/fast_whisper_server.sh"
start_time=$(date +'%m-%d-%Y-%H-%M-%S')
docker run --name "whisper-service_$start_time" -p $PORT:$PORT --gpus all -dit label/whisper-server:0.0.1 $script_command "--port $PORT --gpu_index $GPU_INDEX --model_size $MODEL_SIZE $ONLINE $ENABLE_CORRECTIONS $VERBOSE"
