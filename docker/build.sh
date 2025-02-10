#!/bin/bash
host_dir=${1:-$(pwd)}
docker build -t aker:latest "$host_dir"/docker