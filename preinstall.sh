C_XLA_URL=https://github.com/joelberkeley/spidr/releases/download/c-xla-$(cat backend/VERSION)/c_xla_extension-x86_64-linux-cpu.tar.gz
curl -s -L $C_XLA_URL | tar xzf - -C /usr/local/lib --strip-components 1
bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && ldconfig
