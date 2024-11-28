#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat XLA_VERSION)"

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
    echo "OS ${osu} not handled"
    exit 1
    ;;
esac

(
  cd spidr/backend
  mkdir xla
  install_xla "$rev" xla
  (cd xla; ./configure.py --backend=cpu --os=$os)
  bazel build //:c_xla
  rm -rf xla
)
mv "spidr/backend/bazel-bin/libc_xla.${ext}" "libc_xla-${os}-${arch}.${ext}"
