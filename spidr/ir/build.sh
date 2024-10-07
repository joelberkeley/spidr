#!/bin/sh -e

#script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
#
#(
#  cd $script_dir
#  bazel build //:ir
#)
#
#mv "$script_dir/bazel-bin/libir.so" libir-linux.so

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev="$(cat STABLEHLO_VERSION)"

mkdir spidr/ir/stablehlo
install_stablehlo "$rev" spidr/ir/stablehlo

(
  cd spidr/ir
  bazel build //:ir
  rm -rf stablehlo
)
mv spidr/ir/bazel-bin/libir.so libir-linux.so
