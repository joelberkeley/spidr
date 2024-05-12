sha_short {
  # https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection
  # > Generally, eight to ten characters are more
  # > than enough to be unique within a project.
  sha="$1"
  echo "${sha:0:10}"
}

install_xla {
  dir=$(mktemp -d)
  (
    cd dir
    git init
    git remote add origin https://github.com/openxla/xla
    git fetch --depth 1 origin $1
    git checkout FETCH_HEAD
  )
  echo $dir
}
