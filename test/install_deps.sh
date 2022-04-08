#!/bin/bash -e

TMP_DIR=$(mktemp -d -p $(pwd))

cd $TMP_DIR

git clone --depth 1 --branch v0.5.0 https://github.com/stefan-hoeck/idris2-elab-util.git
idris2 --install idris2-elab-util/elab-util.ipkg
git clone --depth 1 --branch v0.5.0 https://github.com/stefan-hoeck/idris2-sop.git
idris2 --install idris2-sop/sop.ipkg
git clone --depth 1 --branch v0.5.0 https://github.com/stefan-hoeck/idris2-pretty-show.git
idris2 --install idris2-pretty-show/pretty-show.ipkg
git clone --depth 1 --branch v0.5.0 https://github.com/stefan-hoeck/idris2-hedgehog.git
idris2 --install idris2-hedgehog/hedgehog.ipkg

cd .. && rm -rf $TMP_DIR
