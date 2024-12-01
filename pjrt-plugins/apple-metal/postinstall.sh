#!/bin/sh -e

if [ "$SPIDR_INSTALL_SUPPORT_LIBS" = false ]; then exit 0; fi

script_dir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."

os="$(uname)"
case $os in
  'Darwin')
    ;;
  *)
    echo "WARNING: OS $os not supported, unable to fetch supporting libraries."
    exit 0
    ;;
esac

prefix=jax_metal-0.1.0-py3-none-macosx_11_0_arm64
curl -fsL "https://files.pythonhosted.org/packages/80/af/ed482a421a868726e7ca3f51ac19b0c9a8e37f33f54413312c37e9056acc/jax_metal-0.1.0-py3-none-macosx_11_0_arm64.whl" \
  -o "$prefix.zip"
unzip "$prefix.zip"
libdir="$(idris2 --libdir)/pjrt-plugin-xla-cpu-0.0.1/lib"
mkdir -p libdir
mv "$prefix/jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib" "$libdir/pjrt_plugin_apple_metal.dylib"
rm -rf "$prefix.zip" $prefix
