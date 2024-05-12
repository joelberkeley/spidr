xla_short_version () {
  # https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection
  # > Generally, eight to ten characters are more
  # > than enough to be unique within a project.
  sha=$(cat XLA_VERSION)
  echo "${sha:0:10}"
}

install_xla () {
  rev=$(cat XLA_VERSION)
  (
    cd $1
    git init
    git remote add origin https://github.com/openxla/xla
    git fetch --depth 1 origin
    git checkout FETCH_HEAD
  )
}
