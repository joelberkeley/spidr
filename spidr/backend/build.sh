#!/bin/bash -e

here=$(dirname "$(readlink -f "$0")")
cd "$here/../.."

. ./dev.sh

mkdir spidr/backend/xla
install_xla spidr/backend/xla

(
  cd spidr/backend
  bazel build //:c_xla
  rm -rf xla
)

mv spidr/backend/bazel-bin/libc_xla.so .
