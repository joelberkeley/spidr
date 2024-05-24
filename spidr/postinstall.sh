#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
c_xla_version=$(cat "$script_dir/backend/VERSION")

bin_uri="https://github.com/joelberkeley/spidr/releases/download/c-xla-v$c_xla_version/libc_xla.so"

if [ "$SPIDR_MANUAL_INSTALL" = 0 ] || [ -z "$SPIDR_MANUAL_INSTALL" ]; then
  curl -sLO "$bin_uri" --create-dirs --output-dir "$(idris2 --libdir)/spidr-0.0.6/lib"
elif [ "$SPIDR_MANUAL_INSTALL" = 1 ]; then
  echo "Idris API installed. Now install supporting libraries to your library path."
  echo ""
  echo "    $bin_uri"
  echo ""
else
  echo "Invalid value $SPIDR_MANUAL_INSTALL found for SPIDR_MANUAL_INSTALL. Expected 0 or 1 if set."
fi
