rm -r /usr/local/lib/xla_extension
rm -r /usr/local/lib/c_xla_extension

rm /etc/ld.so.conf.d/xla_extension.conf
rm /etc/ld.so.conf.d/c_xla_extension.conf

ldconfig
