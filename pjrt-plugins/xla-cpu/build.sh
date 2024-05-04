set -e

../../install_xla.sh
cd xla
./configure.py --backend=CPU
bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
mv bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so ../pjrt_plugin_xla_cpu.so
cd ..
