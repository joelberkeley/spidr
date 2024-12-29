FROM ubuntu:24.04 as runtime

RUN apt-get update && apt-get install -y git curl
RUN curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    cuda-toolkit-12-6 \
    cudnn9-cuda-12 \
    libnccl2=2.23.4-1+cuda12.6

FROM runtime as devel-xla
COPY --from=runtime / /

RUN apt-get install -y clang lld
RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-amd64.deb
RUN apt-get install -y ./bazelisk-amd64.deb

FROM runtime as devel-idris
COPY --from=runtime / /

RUN apt-get update && apt-get install -y git libgmp3-dev build-essential chezscheme
RUN git clone https://github.com/stefan-hoeck/idris2-pack.git

RUN cd idris2-pack && make micropack SCHEME=chezscheme

ENV PATH=$PATH:/root/.pack/bin/
