#!/bin/bash
# Build Docker image
DockerImageName="anomalygpt"

docker build -t ${DockerImageName} .

echo "\nImage built. Starting container..."
# Run the container with GPU support and mount checkpoints
# Map port 7860 for gradio

docker run --rm -it --gpus all -p 7860:7860 \
    -v $(pwd)/pretrained_ckpt:/app/pretrained_ckpt \
    -v $(pwd)/code/ckpt:/app/code/ckpt \
    -v $(pwd)/data:/app/data \
    ${DockerImageName}
