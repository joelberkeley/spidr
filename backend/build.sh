curl -s -L https://github.com/elixir-nx/xla/releases/download/v$(cat XLA_EXT_VERSION)/xla_extension-$1.tar.gz | tar xz
BAZEL_CXXOPTS='-std=c++14' bazel-6.4.0 build //:c_xla_extension
