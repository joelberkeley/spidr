#!/bin/bash -e

TMP_DIR=$(mktemp -d -p $(pwd))

cd $TMP_DIR

git clone https://github.com/Z-snails/Idris2-hashable.git
(cd Idris2-hashable; git checkout d97d2c39d9199941e2de1991224f564fc4b956dd)
idris2 --install Idris2-hashable/hashable.ipkg

cd - && rm -rf $TMP_DIR
