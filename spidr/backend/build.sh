set -e

. ./dev.sh

dir="$(dirname "$(readlink -f "$0")")"

(
  cd dir
  mkdir xla
  install_xla xla
  bazel build //:c_xla
  rm -rf xla
)

mv dir/bazel-bin/libc_xla.so .
