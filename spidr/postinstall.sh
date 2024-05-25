#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
xla_ext_version=$(cat "$script_dir/backend/VERSION")

os=$(uname)
if [ "$os" != "Linux" ]; then
  echo "OS ${os} not supported, unable to fetch supporting libraries."
  exit 0;
fi;

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/xla-v$xla_ext_version/libxla-linux.so" \
  -o libxla.so --create-dirs --output-dir "$(idris2 --libdir)/spidr-0.0.6/lib"
