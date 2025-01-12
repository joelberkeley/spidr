#!/bin/sh -e

if [ "$SPIDR_INSTALL_SUPPORT_LIBS" = false ]; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
. ./dev.sh
rev=$(cat XLA_VERSION)
cd - > /dev/null 2>&1

os="$(uname)"
case $os in
  'Linux')
    platform=linux-x86_64
    ext=so
    ;;
  'Darwin')
    platform=darwin-aarch64
    ext=dylib
    ;;
  *)
    echo "WARNING: OS $os not supported, unable to fetch supporting libraries."
    exit 0
    ;;
esac

curl -fsL "https://github.com/joelberkeley/spidr/releases/download/xla-$(short_revision "$rev")/pjrt_plugin_xla_cpu-$platform.$ext" \
  -o "pjrt_plugin_xla_cpu.${ext}" --create-dirs --output-dir "$(idris2 --libdir)/pjrt-plugin-xla-cpu-0.0.1/lib"
