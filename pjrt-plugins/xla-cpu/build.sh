set -e

source dev.sh
xla_dir=$(mktemp -d)
install_xla $xla_dir
(
  cd $xla_dir
  ./configure.py --backend=CPU
  bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
)
mv $xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so pjrt_plugin_xla_cpu.so
rm -rf $xla_dir
