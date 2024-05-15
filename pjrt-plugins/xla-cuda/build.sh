set -e

. ./dev.sh

# is this check enough?
# what if we rename the plugin?
# what if the idris code changes?
# anything else?

# seeing error
# error: Could not access 'HEAD^'
# in github
if [ -z "$(git diff --exit-code HEAD^ XLA_VERSION)" ]; then
  wget "https://github.com/joelberkeley/spidr/releases/download/xla-$(xla_short_version)/pjrt_plugin_xla_cuda.so"
  exit 0;
fi

xla_dir=$(mktemp -d)
install_xla $xla_dir
(
  cd $xla_dir
  # note we're not using ./configure.py, see https://github.com/openxla/xla/issues/12017
  bazel build --config release_gpu_linux //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
)
mv $xla_dir/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so pjrt_plugin_xla_cpu.so
rm -rf $xla_dir
