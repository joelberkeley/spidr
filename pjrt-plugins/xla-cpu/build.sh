#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)

osu="$(uname)"
case $osu in
  'Linux')
    os='linux'
    bin_ext=".so"
    bazel_build_args=""
    ;;
  'Darwin')
    os='darwin'
    bin_ext=".dylib"
    bazel_build_args="--config=macos_arm64"
    ;;
  *)
    echo "OS ${osu} not handled"
    exit 1
    ;;
esac

xla_dir=$(mktemp -d)
install_xla "$rev" "$xla_dir"
(
  cd "$xla_dir"
  ./configure.py --backend=CPU --os=$os
  bazel build $bazel_build_args //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv "$xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" "pjrt_plugin_xla_cpu-${os}${bin_ext}"
