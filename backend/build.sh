curl -fL https://github.com/elixir-nx/xla/releases/download/v$(cat XLA_EXT_VERSION)/xla_extension-x86_64-linux-gnu-$1.tar.gz | tar xz
BAZEL_CXXOPTS='-std=c++14' ./bazelisk build //:c_xla_extension
