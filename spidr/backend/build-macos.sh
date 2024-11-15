#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat XLA_VERSION)"

(
  cd spidr/backend
#  mkdir xla
#  install_xla "$rev" xla
  (cd xla; ./configure.py --backend=cpu --os=darwin)
  bazel build //:c_xla
  rm -rf xla
)
#mv spidr/backend/bazel-bin/libc_xla.so libc_xla-linux.so
