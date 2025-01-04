#!/bin/sh -e

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/.."

export SPIDR_INSTALL_SUPPORT_LIBS=false

res=0

for f in tutorials/*.ipkg; do
  pack --no-prompt typecheck "$f" || res=$?
done

exit $res
