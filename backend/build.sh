mkdir xla
cd xla || exit
git init
git remote add origin https://github.com/openxla/xla
git fetch origin "$(cat "$XLA_REVISION")"
git reset --hard FETCH_HEAD
./configure.py --backend=CPU
cd - || exit

# maybe --logging [4, 5 or 6]
BAZEL_CXXOPTS=-std=c++17 bazel build --verbose_failures --experimental_repo_remote_exec //:c_xla
