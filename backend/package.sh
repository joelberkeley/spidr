DIR=$(mktemp -d)
mkdir -p $DIR/$1/c_xla_extension/lib
mv bazel-bin/libc_xla_extension.so $DIR/$1/c_xla_extension/lib
tar cfz $1.tar.gz -C $DIR .
