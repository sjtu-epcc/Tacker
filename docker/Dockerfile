FROM docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && \
    echo "Asia\nShanghai" | apt-get install -y tzdata && \
    apt-get install --reinstall zlibc zlib1g zlib1g-dev && \
    apt-get install libffi-dev libssl-dev libreadline-dev -y && \
    apt-get install -y \
    apt-utils \
    git \
    curl \
    ca-certificates \
    dirmngr \
    gpg \
    wget \
    vim \
    nano \
    zsh \
    autojump \
    screen \
    tmux \
    cmake \
    build-essential nghttp2 libnghttp2-dev libssl-dev && \
    apt-get clean

# Install Python3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -zxf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --prefix=/usr/local/python3 && \
    ./configure --enable-optimizations && \
    make -j8 && \
    make install && \
    rm -rf /usr/bin/python3 && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python3 && \
    cd .. && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz

# Install boost json
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.84.0/source/boost_1_84_0.tar.gz && \
    tar -zxvf boost_1_84_0.tar.gz && \
    cd boost_1_84_0  && \
    ./bootstrap.sh --with-libraries=json && \
    ./b2 && \
    ./b2 install