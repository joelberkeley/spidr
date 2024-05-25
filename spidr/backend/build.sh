#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat XLA_VERSION)"

mkdir spidr/backend/xla
install_xla "$rev" spidr/backend/xla

(
  cd spidr/backend
  bazel build //:xla
  rm -rf xla
)
mv spidr/backend/bazel-bin/libc_xla.so libxla-linux.so
