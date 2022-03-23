#!/bin/bash -e

TMP_DIR=$(mktemp -d -p $(pwd))

cd $TMP_DIR

git clone --depth 1 https://github.com/Z-snails/Idris2-hashable.git
idris2 --install Idris2-hashable/hashable.ipkg

cd .. && rm -rf $TMP_DIR
