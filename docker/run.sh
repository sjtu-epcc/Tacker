#!/bin/bash
host_dir=${1:-$(pwd)}
docker run --name aker -d -it -v host_dir:/workspace/aker --gpus all --privileged=true aker:latest