#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat XLA_VERSION)"

osu="$(uname)"
case $osu in
  'Linux')
    os='linux'
    bin_ext=".so"
    cxxopt="-std=c++17"
    ;;
  'Darwin')
    os='darwin'
    bin_ext=".dylib"
    cxxopt="-std=c++1z"
    ;;
  *)
    echo "OS ${osu} not handled"
    exit 1
    ;;
esac

(
  cd spidr/backend
  mkdir xla
  install_xla "$rev" xla
  (cd xla; ./configure.py --backend=cpu --os=$os)
  bazel build --cxxopt=${cxxopt} //:c_xla
  rm -rf xla
)
mv "spidr/backend/bazel-bin/libc_xla${bin_ext}" "libc_xla-${os}${bin_ext}"
