FROM nvcr.io/nvidia/tensorrt:23.11-py3

RUN apt-get update && apt-get install -y git curl libgmp3-dev build-essential chezscheme
RUN git clone https://github.com/stefan-hoeck/idris2-pack.git

RUN cd idris2-pack && make micropack SCHEME=chezscheme

ENV PATH=$PATH:/root/.pack/bin/
