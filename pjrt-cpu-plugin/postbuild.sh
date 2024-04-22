C_XLA_VERSION=$(cat backend/VERSION)

echo "
PJRT CPU plugin Idris API installed. Now install the plugin binary with

    sudo curl -s -L https://github.com/joelberkeley/spidr/releases/download/-v$C_XLA_VERSION/xla-x86_64-linux-$1.so /usr/local/lib/spidr/libc_xla.so
    echo "/usr/local/lib/spidr/" | sudo tee -a /etc/ld.so.conf.d/spidr.conf
    sudo ldconfig

When you uninstall this plugin, you may wish to remove installed artifacts

    /etc/ld.so.conf.d/spidr.conf
    /usr/local/lib/spidr/

and update shared library links with

    sudo ldconfig
"
