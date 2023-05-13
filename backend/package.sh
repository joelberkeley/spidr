DIR=$(mktemp -d)
mkdir -p $DIR/$1/lib
mv bazel-bin/libc_xla_extension.so $DIR/$1/lib
tar cfz $1.tar.gz -C $DIR .
