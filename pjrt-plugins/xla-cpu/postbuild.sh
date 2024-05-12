SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
XLA_VERSION=$(cat "$SCRIPT_DIR"/../../XLA_VERSION)

echo "
PJRT XLA CPU plugin Idris API installed. Now install the plugin binary with

    sudo curl -s -L ??? /usr/local/lib/pjrt-plugins/pjrt_xla_cpu_plugin.so
    echo "/usr/local/lib/pjrt-plugins/" | sudo tee -a /etc/ld.so.conf.d/pjrt-plugins.conf
    sudo ldconfig

When you uninstall this plugin, you may wish to installed references and artifacts at

    /etc/ld.so.conf.d/pjrt-plugins.conf
    /usr/local/lib/pjrt-plugins/

and update shared library links with

    sudo ldconfig
"
