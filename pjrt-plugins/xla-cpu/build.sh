#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)

cd "$script_dir"

mkdir xla
install_xla "$xla_rev" xla
(
  cd xla
  ./configure.py --backend=CPU
  bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so pjrt_plugin_xla_cpu-linux.so

rm -rf xla
