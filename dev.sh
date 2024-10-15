#!/bin/sh -e

short_revision () {
  # https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection
  # > Generally, eight to ten characters are more
  # > than enough to be unique within a project.
  rev=$1
  echo "${rev%%"${rev##??????????}"}"
}

install_repository () {
  if [ -z "$2" ]; then
    echo "Usage: install_$3 <revision> <install-path>."
    exit 1;
  fi

  if [ "$(ls -A "$2")" ]; then
    echo "Directory at path $2 is not empty, refusing to install $3 to this directory."
    exit 1;
  fi

  (
    cd "$2"
    git init
    git remote add origin "https://github.com/openxla/$3"
    git fetch --depth 1 origin "$1"
    git checkout FETCH_HEAD
  )
}

install_xla () {
  install_repository $1 $2 "xla"
}

install_stablehlo () {
  install_repository $1 $2 "stablehlo"
}
