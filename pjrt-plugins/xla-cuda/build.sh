set -e

source dev.sh
xla_dir=$(mktemp -d)
install_xla $xla_dir
(
  cd $xla_dir
  # note we're not using ./configure.py, see https://github.com/openxla/xla/issues/12017
  bazel build --config release_gpu_linux //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
)
mv $xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so pjrt_plugin_xla_cpu.so
rm -rf $xla_dir
