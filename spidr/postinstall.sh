#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
c_xla_version=$(cat "$script_dir/backend/VERSION")

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/c-xla-$c_xla_version/libc_xla.so"
