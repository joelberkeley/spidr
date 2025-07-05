#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
xla_rev="$(cat XLA_VERSION)"
enzyme_rev="$(cat spidr/backend/ENZYME_JAX_VERSION)"

(
  cd spidr/backend

  mkdir xla
  install_xla "$xla_rev" xla
  (cd xla; ./configure.py --backend=cpu)

  # depending on Enzyme-JAX is problematic as it fixes the XLA version. Can we only depend on enzyme?
  # seems unlikely that they could decouple XLA entirely. They almost certainly can't decouple stablehlo
  mkdir Enzyme-JAX
  install_enzyme "$enzyme_rev" Enzyme-JAX
  patch Enzyme-JAX/src/enzyme_ad/jax/BUILD < BUILD.patch

  bazel build //:c_xla
  rm -rf xla Enzyme-JAX
)

mv spidr/backend/bazel-bin/libc_xla.so libc_xla-linux.so
