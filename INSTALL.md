# Installation

## Install the Idris frontend

1. Install the Idris2 package manager [pack](https://github.com/stefan-hoeck/idris2-pack).
2. Install spidr with
   ```
   pack install spidr
   ```

## Install the XLA backend

### CPU only

If you intend to run spidr on GPU, skip to the [GPU instructions](#gpu)

3. Install XLA
   ```bash
   wget https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-cpu.tar.gz -qO- | sudo tar -C /usr/local/lib -xvz ./xla_extension/lib/libxla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/xla_extension/lib" >> /etc/ld.so.conf.d/xla_extension.conf' && sudo ldconfig
   ```
4. Install the C interface to XLA
   ```bash
   sudo wget https://github.com/joelberkeley/spidr/releases/download/v0.0.5/libc_xla_extension.so -P /usr/local/lib/c_xla_extension/lib
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && sudo ldconfig
   ```

### GPU

If you do *not* intend to run spidr on GPU, skip this section.

3. Install the NVIDIA prerequisites for running TensorFlow on GPU, as listed on the TensorFlow GPU [installation page](https://www.tensorflow.org/install/gpu). **There is no need to install TensorFlow itself**.
4. Install XLA
   ```bash
   wget https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-cuda111.tar.gz -qO- | sudo tar -C /usr/local/lib -xvz ./xla_extension/lib/libxla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/xla_extension/lib" >> /etc/ld.so.conf.d/xla_extension.conf' && sudo ldconfig
   ```
5. Install the C interface to XLA
   ```bash
   sudo wget https://github.com/joelberkeley/spidr/releases/download/v0.0.5/libc_xla_extension.so -P /usr/local/lib/c_xla_extension/lib
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && sudo ldconfig
   ```
