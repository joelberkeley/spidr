script_dir="$(dirname "$(readlink -f "$0")")"
c_xla_version=$(cat "$script_dir/backend/VERSION")

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/c-xla-$(c_xla_version)/libc_xla.so"
