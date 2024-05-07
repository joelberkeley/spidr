set -e

../../install_xla.sh
cd xla
./configure.py --backend=CPU
bazel build --nobuild_runfile_links --nolegacy_external_runfiles //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
mv bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so ../pjrt_plugin_xla_cpu.so
cd ..
