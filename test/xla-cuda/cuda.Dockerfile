FROM ubuntu:24.04

RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# this is a pain - build artifacts assume cuda 12.5 (are we sure?), but there's no libnccl for 12.5 on ubuntu 24.04
# (note the tests pass with libnccl 12.6 with cuda 12.5, but this might just be because we're not using any collective communication yet)
# so either we build artifacts for cuda 12.6, risk using 12.5 with 12.6, or regress to ubuntu 22.04
# I'd prefer to use 12.6 and 24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    cuda-toolkit-12-6 \
    cudnn-cuda-12 \
    libnccl2=2.22.3-1+cuda12.6
