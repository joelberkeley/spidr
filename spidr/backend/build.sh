#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
xla_rev="$(cat XLA_VERSION)"
enzyme_rev="$(cat spidr/backend/ENZYME_JAX_VERSION)"

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

(
  cd spidr/backend
  mkdir xla
  install_xla "$xla_rev" xla
  (cd xla; ./configure.py --backend=cpu --os=$os)
  mkdir Enzyme-JAX
  install_enzyme "$enzyme_rev" Enzyme-JAX
  # sed -i -e 's/"-Werror=unused-variable",//g' Enzyme-JAX/src/enzyme_ad/jax/BUILD
  bazel build //:c_xla
  rm -rf xla
)
mv "spidr/backend/bazel-bin/libc_xla.$ext" "libc_xla-$os-$arch.$ext"
