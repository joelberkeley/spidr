set -e

. ./dev.sh

(
  cd spidr/backend
  mkdir xla
  install_xla xla
  bazel build //:c_xla
  rm -rf xla
)

mv spidr/backend/bazel-bin/libc_xla.so .
