#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)
cd - > /dev/null 2>&1

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/xla-$(short_revision "$rev")/pjrt_plugin_xla_cpu.so" \
  --create-dirs --output-dir "$(idris2 --libdir)/pjrt-plugin-xla-cpu-0.0.1/lib"
