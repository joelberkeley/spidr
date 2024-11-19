#!/bin/sh -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  os="linux"
  bin_ext=".so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  os="darwin"
  bin_ext=".dylib"
else
  echo "OS ${OSTYPE} not handled, expected linux-gnu or darwin"
  exit 1
fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)

xla_dir=$(mktemp -d)
install_xla "$rev" "$xla_dir"
(
  cd "$xla_dir"
  ./configure.py --backend=CPU --os=$os
  bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv "$xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" "pjrt_plugin_xla_cpu-${os}${bin_ext}"
