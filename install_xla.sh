if [ -d xla ]; then
  echo "Directory xla already exists, aborting"
  exit 1;
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
COMMIT_SHA="$(cat $SCRIPT_DIR/XLA_COMMIT_SHA)"

mkdir xla
cd xla
git init
git remote add origin https://github.com/openxla/xla
git fetch --depth 1 origin $COMMIT_SHA
git checkout FETCH_HEAD
cd -
