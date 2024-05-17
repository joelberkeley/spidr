here="$(dirname "$(readlink -f "$0")")"
cwd=$pwd
cd $here
cd ../..
. ./dev.sh
cd $cwd

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/xla-$(xla_short_version)/pjrt_plugin_xla_cuda.so"
