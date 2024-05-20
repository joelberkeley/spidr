#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."

. ./dev.sh
rev="$(cat XLA_VERSION)"

xla_dir=$(mktemp -d)
install_xla "$rev" "$xla_dir"
(
  cd "$xla_dir"
  # note we're not using `./configure.py --backend=CUDA` as it requires a GPU, but the build
  # itself doesn't, see https://github.com/openxla/xla/issues/12017
  bazel build --config release_gpu_linux //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
)
mv "$xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so" pjrt_plugin_xla_cuda.so
