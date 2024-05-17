#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."

. ./dev.sh

mkdir spidr/backend/xla
install_xla spidr/backend/xla

(
  cd spidr/backend
  bazel build //:c_xla
  rm -rf xla
)

mv spidr/backend/bazel-bin/libc_xla.so .
