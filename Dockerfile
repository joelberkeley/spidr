FROM ghcr.io/stefan-hoeck/idris2-pack:nightly-240430 as pack

COPY . /spidr

WORKDIR spidr/test/xla-cuda

RUN pack --no-prompt build xla-cuda.ipkg

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y cuda-toolkit-12-3
COPY *.deb .
# install appropriate deb from cuDNN archives
RUN apt-get install ./cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

RUN apt-get install -y chezscheme

WORKDIR spidr/test

COPY test/*.so .
COPY --from=pack /spidr/test/xla-cuda/build/ build/

CMD ./build/exec/test
