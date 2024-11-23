#!/bin/sh -e

if [ "$SPIDR_INSTALL_SUPPORT_LIBS" = false ]; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)
cd - > /dev/null 2>&1

os=$(uname)
if [ "$os" != "Linux" ]; then
  echo "OS ${os} not supported, unable to fetch supporting libraries."
  exit 0;
fi;

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/xla-$(short_revision "$rev")/pjrt_plugin_xla_cuda-linux.so" \
  -o pjrt_plugin_xla_cuda.so --create-dirs --output-dir "$(idris2 --libdir)/pjrt-plugin-xla-cuda-0.0.1/lib"
