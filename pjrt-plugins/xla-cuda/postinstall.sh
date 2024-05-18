#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"/../..
. ./dev.sh
rev=$(cat XLA_VERSION)
cd - > /dev/null 2>&1

bin_uri="https://github.com/joelberkeley/spidr/releases/download/xla-$(short_revision "$rev")/pjrt_plugin_xla_cuda.so"

if [ "$SPIDR_MANUAL_INSTALL" = 0 ] || [ -z "$SPIDR_MANUAL_INSTALL" ]; then
  curl -sLO "$bin_uri"
elif [ "$SPIDR_MANUAL_INSTALL" = 1 ]; then
  echo "Idris API installed. Now install supporting libraries to your library path."
  echo ""
  echo "    $bin_uri"
  echo ""
else
  echo "Invalid value $SPIDR_MANUAL_INSTALL found for SPIDR_MANUAL_INSTALL. Expected 0 or 1 if set."
fi
