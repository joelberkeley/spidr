PREFIX=$2
PLATFORM=$1

DIR=$(mktemp -d)
mkdir -p $DIR/$PREFIX-$PLATFORM/lib
mv bazel-bin/libc_xla_extension.so $DIR/$PREFIX-$PLATFORM/lib
tar cfz $PREFIX-$PLATFORM.tar.gz -C $DIR .
