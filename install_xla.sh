if [ -d xla ]; then
  echo "Directory xla already exists, aborting"
  exit 1;
fi

git init
git remote add origin https://github.com/openxla/xla
git fetch origin "$(cat ../XLA_COMMIT_SHA)"
git reset --hard FETCH_HEAD
