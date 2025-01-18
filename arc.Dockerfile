FROM ghcr.io/stefan-hoeck/idris2-pack:noble AS build

COPY . /spidr

WORKDIR /spidr

RUN SPIDR_INSTALL_SUPPORT_LIBS=false pack --no-prompt build test/xla-cuda/xla-cuda.ipkg

FROM xla-cuda

RUN apt update && apt install -y chezscheme curl

COPY --from=build /spidr/test/xla-cuda/build/exec /xla-cuda
COPY libc_xla.so /xla-cuda/libc_xla.so
COPY pjrt_plugin_xla_cuda.so /xla-cuda/pjrt_plugin_xla_cuda.so

ENV LD_LIBRARY_PATH=/xla-cuda

# singularity starts a container at $HOME regardless of any WORKDIR directives,
# so I need `singularity run --pwd /xla-cuda` for this to work

CMD ["./test"]
