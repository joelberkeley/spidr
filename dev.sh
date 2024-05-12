xla_short_version () {
  # https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection
  # > Generally, eight to ten characters are more
  # > than enough to be unique within a project.
  ls
  rev=$(cat XLA_VERSION)
  echo ${rev%%"${rev##??????????}"}
}

install_xla () {
  if [ -z $1 ]; then
    echo "Usage: install_xla <path>."
    exit 1;
  fi

  if [ $(ls -A $1) ]; then
    echo "Directory at path $1 is not empty, refusing to install XLA to this directory."
    exit 1;
  fi

  ls
  rev=$(cat XLA_VERSION)
  (
    cd $1
    git init
    git remote add origin https://github.com/openxla/xla
    git fetch --depth 1 origin $rev
    git checkout FETCH_HEAD
  )
}
