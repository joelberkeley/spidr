# Installation

**Note:** these instructions may have changed since the latest release. Navigate to the instructions at the latest git tag for accurate details.

## Install the Idris frontend

1. Clone or download the spidr source code at tag v0.0.5 with, for example,
   ```bash
   git clone --depth 1 --branch v0.0.5 https://github.com/joelberkeley/spidr.git
   ```
   See the git history for installing earlier versions.
2. [Install Idris 2](https://github.com/idris-lang/Idris2/blob/main/INSTALL.md). We recommend using [Homebrew](https://brew.sh/) if you're unsure which option to use.
3. Install spidr's Idris dependencies by running `./install_deps.sh` located in the spidr root directory.
4. In the spidr root directory, install spidr with
   ```bash
   idris2 --install spidr.ipkg
   ```
5. When building Idris code that depends on spidr, you will need to include both `spidr` and its dependencies `hashable` and `contrib`. Either use the `-p` command line flag, or add these dependencies to your project's configuration e.g. your .ipkg `depends` field.

## Install the XLA backend

### CPU only

If you intend to run spidr on GPU, skip to the [GPU instructions][#gpu]

6. Install XLA with
   ```
   $ wget https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-cpu.tar.gz -qO- | sudo tar -C /usr/local/lib -xvz ./xla_extension/lib/libxla_extension.so --strip-components=3
   ```
7. Install the C interface to XLA with
   ```
   $ sudo wget https://github.com/joelberkeley/spidr/releases/download/v0.0.5/libc_xla_extension.so -P /usr/local/lib
   ```
8. Update linker caches with
   ```
   $ sudo ldconfig /usr/local/lib
   ```

### GPU

If you do *not* intend to run spidr on GPU, skip this section.

6. Install the NVIDIA prerequisites for running TensorFlow on GPU, as listed on the TensorFlow GPU [installation page](https://www.tensorflow.org/install/gpu). **There is no need to install TensorFlow itself**.
7. Install XLA with
   ```
   $ wget https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-cuda111.tar.gz -qO- | sudo tar -C /usr/local/lib -xvz ./xla_extension/lib/libxla_extension.so --strip-components=3
   ```
8. Install the C interface to XLA with
   ```
   $ sudo wget https://github.com/joelberkeley/spidr/releases/download/v0.0.5/libc_xla_extension.so -P /usr/local/lib
   ```
9. Update linker caches with
   ```
   $ sudo ldconfig /usr/local/lib
   ```
