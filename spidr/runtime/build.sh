#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat XLA_VERSION)"

mkdir spidr/runtime/xla
install_xla "$rev" spidr/runtime/xla

(
  cd spidr/runtime
  bazel build //:runtime
  rm -rf xla
)
mv spidr/runtime/bazel-bin/libpjrt.so libpjrt-linux.so
