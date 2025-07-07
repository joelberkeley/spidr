#!/bin/sh -e

install_git_repository () {
  if [ "$(ls -A "$2")" ]; then
    echo "Directory at path $2 is not empty, refusing to install to this directory."
    exit 1;
  fi

  (
    cd "$2"
    git init
    git remote add origin "$3"
    git fetch --depth 1 origin "$1"
    git checkout FETCH_HEAD
  )
}

install_xla () {
  install_git_repository "$1" "$2" https://github.com/openxla/xla
}

install_enzyme () {
  install_git_repository "$1" "$2" https://github.com/EnzymeAD/Enzyme-JAX.git
}
