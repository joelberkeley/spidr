FROM ghcr.io/stefan-hoeck/noble as build

COPY . /spidr

WORKDIR /spidr

RUN pack --no-prompt build test/xla-cuda/xla-cuda.ipkg

FROM xla-cuda

# does this copy over shared libs?
COPY --from=build /spidr/test/xla-cuda/build/exec /spidr/test/xla-cuda/build/exec

RUN apt install -y chezscheme

WORKDIR /spidr/test/xla-cuda/build/exec

CMD ./test
