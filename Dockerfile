FROM ghcr.io/stefan-hoeck/idris2-pack:nightly-240430 as pack

COPY . /spidr

WORKDIR spidr/test/xla-cuda

RUN pack --no-prompt build xla-cuda.ipkg

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y cuda-toolkit-12-3 libcudnn8 libnvinfer8 libnvinfer-plugin8

RUN apt-get install -y chezscheme

WORKDIR spidr/test

COPY test/*.so .
COPY --from=pack /spidr/test/xla-cuda/build/ build/

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$(pwd)

CMD ./build/exec/test
