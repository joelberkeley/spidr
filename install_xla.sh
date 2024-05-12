if [ -d xla ]; then
  echo "Directory xla already exists, aborting"
  exit 1;
fi

script_dir="$(dirname "$(readlink -f "$0")")"
sha="$(cat $script_dir/XLA_VERSION)"

mkdir xla
cd xla
git init
git remote add origin https://github.com/openxla/xla
git fetch --depth 1 origin $sha
git checkout FETCH_HEAD
cd -
