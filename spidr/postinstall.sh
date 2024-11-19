#!/bin/sh -e

if ! $SPIDR_INSTALL_SUPPORT_LIBS; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
xla_ext_version=$(cat "$script_dir/backend/VERSION")

if ! [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
  echo "OS ${OSTYPE} not supported, unable to fetch supporting libraries"
  exit 1
fi

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/c-xla-v$xla_ext_version/libc_xla-linux.so" \
  -o libc_xla.so --create-dirs --output-dir "$(idris2 --libdir)/spidr-0.0.6/lib"
