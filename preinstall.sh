curl -OL https://github.com/joelberkeley/spidr/releases/download/c-xla-$(cat backend/VERSION)/c_xla_extension-x86_64-linux-cpu.tar.gz
sudo tar xzf c_xla_extension-x86_64-linux-cpu.tar.gz -C /usr/local/lib --strip-components 1
sudo bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && sudo ldconfig
