set -e
set -o pipefail

apt update && apt install -y curl build-essential
curl -fL https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-arm64 > bazelisk
chmod +x bazelisk
