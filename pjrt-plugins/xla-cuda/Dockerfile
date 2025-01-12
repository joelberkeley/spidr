FROM ubuntu:24.04

RUN apt-get update && apt-get install -y chezscheme git curl
RUN curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    cuda-toolkit-12-6 \
    cudnn9-cuda-12 \
    libnccl2=2.23.4-1+cuda12.6
