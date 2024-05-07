set -e

../../install_xla.sh
cd xla
# note we're not using ./configure.py, see https://github.com/openxla/xla/issues/12017
bazel build --nobuild_runfile_links --nolegacy_external_runfiles --config release_gpu_linux //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
mv bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so ../pjrt_plugin_xla_cuda.so
cd ..
