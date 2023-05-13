curl -s -L https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-$1.tar.gz | tar xz
BAZEL_CXXOPTS='-std=c++14' bazel build //:c_xla_extension
