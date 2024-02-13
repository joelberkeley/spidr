XLA_EXT_VERSION=$(cat backend/XLA_EXT_VERSION)
C_XLA_EXT_VERSION=$(cat backend/VERSION)

install_intructions() {
  cat <<- EOF
    curl -s -L https://github.com/elixir-nx/xla/releases/download/v$XLA_EXT_VERSION/xla_extension-x86_64-linux-$1.tar.gz | sudo tar xz -C /usr/local/lib ./xla_extension/lib/libxla_extension.so
    curl -s -L https://github.com/joelberkeley/spidr/releases/download/c-xla-v$C_XLA_EXT_VERSION/c_xla_extension-x86_64-linux-$1.tar.gz | sudo tar xz -C /usr/local/lib ./c_xla_extension/lib/libc_xla_extension.so
    echo "/usr/local/lib/xla_extension/lib" | sudo tee -a /etc/ld.so.conf.d/xla_extension.conf
    echo "/usr/local/lib/c_xla_extension/lib" | sudo tee -a /etc/ld.so.conf.d/c_xla_extension.conf
    sudo ldconfig
EOF
}

echo "
spidr Idris API installed. Now install either the CPU or CUDA backend. The following instructions are for Ubuntu Linux. To install the CPU backend, run

$(install_intructions cpu)

Or, to install the CUDA backend, install the NVIDIA prerequisites for running TensorFlow on GPU, as listed on the TensorFlow GPU installation page (there is no need to install TensorFlow itself)

    https://www.tensorflow.org/install/pip

then run

$(install_intructions cuda111)

When you uninstall spidr, you may wish to remove installed XLA artifacts

    /etc/ld.so.conf.d/xla_extension.conf
    /etc/ld.so.conf.d/c_xla_extension.conf
    /usr/local/lib/xla_extension/
    /usr/local/lib/c_xla_extension/

and update shared library links with

    sudo ldconfig
"
