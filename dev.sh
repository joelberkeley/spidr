#!/bin/sh -e

install_xla () {
  if [ -z "$2" ]; then
    echo "Usage: install_xla <xla-revision> <install-path>."
    exit 1;
  fi

  if [ "$(ls -A "$2")" ]; then
    echo "Directory at path $2 is not empty, refusing to install XLA to this directory."
    exit 1;
  fi

  (
    cd "$2"
    git init
    git remote add origin https://github.com/openxla/xla
    git fetch --depth 1 origin "$1"
    git checkout FETCH_HEAD
  )
}
