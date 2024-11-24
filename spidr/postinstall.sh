#!/bin/sh -e

if [ "$SPIDR_INSTALL_SUPPORT_LIBS" = false ]; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
xla_ext_version=$(cat "$script_dir/backend/VERSION")

os="$(uname)"
case $os in
  'Linux')
    ;;
  'Darwin')
    ;;
  *)
    echo "WARNING: OS ${os} not supported, unable to fetch supporting libraries."
    exit 0
    ;;
esac

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/c-xla-v$xla_ext_version/libc_xla-linux.so" \
  -o libc_xla.so --create-dirs --output-dir "$(idris2 --libdir)/spidr-0.0.6/lib"
