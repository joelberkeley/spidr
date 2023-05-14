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
   curl -s -L https://github.com/elixir-nx/xla/releases/download/v$(cat XLA_EXT_VERSION)/xla_extension-x86_64-linux-cpu.tar.gz | sudo tar xz -C /usr/local/lib ./xla_extension/lib/libxla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/xla_extension/lib" >> /etc/ld.so.conf.d/xla_extension.conf' && sudo ldconfig
   ```
4. Install the C interface to XLA
   ```bash
   curl -s -L https://github.com/joelberkeley/spidr/releases/download/c-xla-$(cat backend/VERSION)/c_xla_extension-x86_64-linux-cpu.tar.gz | sudo tar xz -C /usr/local/lib ./c_xla_extension/lib/libc_xla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && sudo ldconfig
   ```

### GPU

If you do *not* intend to run spidr on GPU, skip this section.

3. Install the NVIDIA prerequisites for running TensorFlow on GPU, as listed on the TensorFlow GPU [installation page](https://www.tensorflow.org/install/gpu). **There is no need to install TensorFlow itself**.
4. Install XLA
   ```bash
   curl -s -L https://github.com/elixir-nx/xla/releases/download/v$(cat XLA_EXT_VERSION)/xla_extension-x86_64-linux-cuda111.tar.gz | sudo tar xz -C /usr/local/lib ./xla_extension/lib/libxla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/xla_extension/lib" >> /etc/ld.so.conf.d/xla_extension.conf' && sudo ldconfig
   ```
5. Install the C interface to XLA
   ```bash
   curl -s -L https://github.com/joelberkeley/spidr/releases/download/c-xla-$(cat backend/VERSION)/c_xla_extension-x86_64-linux-cuda111.tar.gz | sudo tar xz -C /usr/local/lib ./c_xla_extension/lib/libc_xla_extension.so
   ```
   ```bash
   sudo bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && sudo ldconfig
   ```
