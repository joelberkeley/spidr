#!/bin/sh -e

os="$(uname)"
case $os in
  'Linux')
    sudo apt-get update && sudo apt-get install -y git libgmp3-dev build-essential chezscheme
    SCHEME=chezscheme
    ;;
  'Darwin')
    brew install chezscheme
    SCHEME=chez
    ;;
  *)
    echo "WARNING: OS $os not supported, unable to install pack."
    exit 1
    ;;
esac

(
  cd "$(mktemp -d)"
  git clone https://github.com/stefan-hoeck/idris2-pack.git
  cd idris2-pack
  make micropack SCHEME=$SCHEME
)
~/.pack/bin/pack switch HEAD
