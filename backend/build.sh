# maybe --logging [4, 5 or 6]
BAZEL_CXXOPTS=-std=c++17 bazel build --verbose_failures --experimental_cc_shared_library --experimental_repo_remote_exec //:c_xla
