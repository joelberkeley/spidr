C_XLA_VERSION=$(cat backend/VERSION)

install_intructions() {
  cat <<- EOF
    sudo curl -s -L https://github.com/joelberkeley/spidr/releases/download/xla-v$C_XLA_VERSION/xla-x86_64-linux-$1.so /usr/local/lib/spidr/libxla.so
    echo "/usr/local/lib/spidr/" | sudo tee -a /etc/ld.so.conf.d/spidr.conf
    sudo ldconfig
EOF
}

echo "
spidr Idris API installed. Now install either the CPU or CUDA backend. To install the CPU backend, run

$(install_intructions cpu)

Or, to install the CUDA backend, install NVIDIA CUDA and cuDNN, then run

$(install_intructions cuda111)

When you uninstall spidr, you may wish to remove installed XLA artifacts

    /etc/ld.so.conf.d/spidr.conf
    /usr/local/lib/spidr/

and update shared library links with

    sudo ldconfig
"
