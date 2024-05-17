#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"/../..
. ./dev.sh
rev=$(cat XLA_VERSION)
cd - >/dev/null 2>&1
rev_short=$(short_revision "$rev")

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/xla-$rev_short/pjrt_plugin_xla_cpu.so"
