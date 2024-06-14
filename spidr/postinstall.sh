#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
xla_ext_version=$(cat "$script_dir/backend/VERSION")

os=$(uname)
if [ "$os" != "Linux" ]; then
  echo "OS ${os} not supported, unable to fetch supporting libraries."
  exit 0;
fi;

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/c-xla-v$xla_ext_version/libc_xla-linux.so" \
  -o libc_xla.so --create-dirs --output-dir "$(idris2 --libdir)/spidr-0.0.6/lib"
