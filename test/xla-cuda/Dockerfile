FROM ghcr.io/stefan-hoeck/idris2-pack:nightly-240430 as pack

COPY . /spidr

WORKDIR spidr/test/xla-cuda

RUN pack --no-prompt build xla-cuda.ipkg

FROM nvcr.io/nvidia/tensorrt:23.11-py3

RUN apt-get update && apt-get install -y chezscheme

WORKDIR spidr/test

COPY test/*.so .
COPY --from=pack /spidr/test/xla-cuda/build/ build/

CMD LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd) ./build/exec/test
