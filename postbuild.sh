C_XLA_VERSION=$(cat backend/VERSION)

install_intructions() {
  cat <<- EOF
    sudo curl -s -L https://github.com/joelberkeley/spidr/releases/download/c-xla-v$C_XLA_VERSION/c_xla-x86_64-linux-$1.so /usr/local/lib/libc_xla.so
    echo "/usr/local/lib/c_xla/" | sudo tee -a /etc/ld.so.conf.d/c_xla.conf
    sudo ldconfig
EOF
}

echo "
spidr Idris API installed. Now install either the CPU or CUDA backend. To install the CPU backend, run

$(install_intructions cpu)

Or, to install the CUDA backend, install NVIDIA CUDA and cuDNN, then run

$(install_intructions cuda111)

When you uninstall spidr, you may wish to remove installed XLA artifacts

    /etc/ld.so.conf.d/c_xla.conf
    /usr/local/lib/c_xla/

and update shared library links with

    sudo ldconfig
"
