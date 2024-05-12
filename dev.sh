xla_short_version () {
  # https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection
  # > Generally, eight to ten characters are more
  # > than enough to be unique within a project.
  sha=$(cat XLA_VERSION)
  echo ${sha%%"${sha##??????????}"}
}

install_xla () {
  if [ -z $1 ]; then
    echo "Directory required as argument, aborting."
  fi

  if [ -d $1 ]; then
    echo "Directory already exists at path $1, aborting."
  fi

  rev=$(cat XLA_VERSION)
  mkdir $1
  (
    cd $1
    git init
    git remote add origin https://github.com/openxla/xla
    git fetch --depth 1 origin
    git checkout FETCH_HEAD
  )
}
