#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."

. ./dev.sh
rev="$(cat XLA_VERSION)"

xla_dir=$(mktemp -d)
install_xla "$(short_revision $rev)" "$xla_dir"
(
  cd "$xla_dir"
  ./configure.py --backend=CPU
  bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv "$xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" pjrt_plugin_xla_cpu.so
