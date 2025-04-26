#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
xla_version=$(cat XLA_VERSION)

osu="$(uname)"
case $osu in
  'Linux')
    os=linux
    arch=x86_64
    ext=so
    ;;
  'Darwin')
    os=darwin
    arch=aarch64
    ext=dylib
    ;;
  *)
    echo "OS $osu not handled"
    exit 1
    ;;
esac

xla_dir=$(mktemp -d)
install_xla "$xla_version" "$xla_dir"
(
  cd "$xla_dir"
  ./configure.py --backend=CPU --os=$os
  bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv "$xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" "pjrt_plugin_xla_cpu-$os-$arch.$ext"
