# Install Bazel
sudo apt update && sudo apt upgrade -y && sudo apt install -y apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install -y bazel

# Build C XLA
sudo apt install -y wget
wget -qO- https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-$1.tar.gz | tar xvz -C backend/
(cd backend; BAZEL_CXXOPTS='-std=c++14' bazel build //:c_xla_extension)
