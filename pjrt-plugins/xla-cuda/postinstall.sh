#!/bin/sh -e

if [ "$SPIDR_LOCAL_INSTALL" = true ]; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
rev=$(cat "$script_dir/VERSION")

os=$(uname)
if [ "$os" != "Linux" ]; then
  echo "OS ${os} not supported, unable to fetch supporting libraries."
  exit 0;
fi;

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/pjrt_plugin_xla_cuda-$rev/pjrt_plugin_xla_cuda-linux.so" \
  -o pjrt_plugin_xla_cuda.so --create-dirs --output-dir "$(idris2 --libdir)/pjrt-plugin-xla-cuda-$rev/lib"
