here="$(dirname "$(readlink -f "$0")")"
c_xla_version=$(cat "$here/backend/VERSION")

curl -sLO "https://github.com/joelberkeley/spidr/releases/download/c-xla-$(c_xla_version)/libc_xla.so"
