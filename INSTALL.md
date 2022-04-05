# Installation

**Note:** these instructions may have changed since the latest release. Navigate to the instructions at the latest git tag for accurate details.

## Install the spidr frontend

1. Clone or download the spidr source code at tag v0.0.4 with, for example,
   ```bash
   git clone --depth 1 --branch v0.0.4 https://github.com/joelberkeley/spidr.git
   ```
   See the git history for installing earlier versions.
2. [Install Idris 2](https://github.com/idris-lang/Idris2/blob/main/INSTALL.md). We recommend using [Homebrew](https://brew.sh/) if you're unsure which option to use.
3. Install spidr's Idris dependencies by running `./install_deps.sh` located in the spidr root directory.
4. In the spidr root directory, install spidr with
   ```bash
   idris2 --install spidr.ipkg
   ```
5. When building your Idris code that depends on spidr, either include `-p spidr` on the command line, or list spidr as a dependency in your project's configuration. In both cases, you will need to include spidr's dependencies with `-p hashable` on the command line or add `hashable` to your project's .ipkg `depends`.

## Install the XLA backend

6. Download the XLA C interface from the [releases page](https://github.com/joelberkeley/spidr/releases), and extract the archive. The extracted directory can be placed anywhere you wish.
7. Download an XLA binary from [elixir-nx/xla](https://github.com/elixir-nx/xla/releases), and extract the archive. See the spidr releases page for details of which versions are supported. Place the directory `xla_extension` into the same path as the directory `c_xla_extension` extracted in step 6.
8. When running code that depends on spidr, you may need to set `LD_LIBRARY_PATH` to include the location of the libc_xla_extension.so shared library located in `c_xla_extension` extracted in step 6.

## Additional steps for execution on GPU

**Note:** We have only tested this on a machine with a single GPU.

9. Install the NVIDIA prerequisites for running TensorFlow on GPU, as listed on the TensorFlow GPU [installation page](https://www.tensorflow.org/install/gpu). **There is no need to install TensorFlow itself**.
