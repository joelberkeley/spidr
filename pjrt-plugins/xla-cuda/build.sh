#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)

cd "$script_dir"
cuda_version=$(cat CUDA_VERSION)
cudnn_version=$(cat CUDNN_VERSION)

cd ../..
. ./dev.sh
xla_rev=$(cat XLA_VERSION)

cd "$script_dir"

mkdir xla
install_xla "$xla_rev" xla
(
  cd xla
  # note we're not using `./configure.py --backend=CUDA` as it requires a GPU, but the build
  # itself doesn't, see https://github.com/openxla/xla/issues/12017
  bazel build \
    --config release_gpu_linux \
    --repo_env HERMETIC_CUDA_VERSION="$cuda_version" \
    --repo_env HERMETIC_CUDNN_VERSION="$cudnn_version" \
    //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
)
mv xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so pjrt_plugin_xla_cuda-linux.so

rm -rf xla
