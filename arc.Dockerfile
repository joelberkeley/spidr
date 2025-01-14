FROM ghcr.io/stefan-hoeck/noble as build

COPY . /spidr

WORKDIR /spidr

RUN SPIDR_INSTALL_SUPPORT_LIBS=false pack --no-prompt build test/xla-cuda/xla-cuda.ipkg

FROM xla-cuda

RUN apt update && apt install -y chezscheme curl

COPY --from=build /spidr/test/xla-cuda/build/exec /xla-cuda

WORKDIR /xla-cuda

RUN curl -LO "https://github.com/joelberkeley/spidr/releases/download/xla-2fb20601f1/pjrt_plugin_xla_cuda-linux-x86_64.so"
RUN curl -LO "https://github.com/joelberkeley/spidr/releases/download/c-xla-v0.0.16/libc_xla-linux-x86_64.so"
RUN mv libc_xla-linux-x86_64.so libc_xla.so
RUN mv pjrt_plugin_xla_cuda-linux-x86_64.so pjrt_plugin_xla_cuda.so

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/xla-cuda
CMD ./test
