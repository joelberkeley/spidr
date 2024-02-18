set -e
set -o pipefail

DIR=$(mktemp -d)
mkdir -p $DIR/c_xla_extension/lib
mv bazel-bin/libc_xla_extension.so $DIR/c_xla_extension/lib
tar cfz c_xla_extension.tar.gz -C $DIR .
