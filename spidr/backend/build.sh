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
rev="$(cat XLA_VERSION)"

(
  cd spidr/backend
  mkdir xla
  install_xla "$rev" xla
  (cd xla; ./configure.py --backend=cpu --os=$os)
  bazel build //:c_xla
  rm -rf xla
)
mv "spidr/backend/bazel-bin/libc_xla${bin_ext}" "libc_xla-${os}${bin_ext}"
