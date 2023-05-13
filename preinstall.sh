XLA_URL=https://github.com/elixir-nx/xla/releases/download/v0.3.0/xla_extension-x86_64-linux-cpu.tar.gz
C_XLA_URL=https://github.com/joelberkeley/spidr/releases/download/c-xla-$(cat backend/VERSION)/c_xla_extension-x86_64-linux-cpu.tar.gz

curl -s -L $XLA_URL | tar xz -C /usr/local/lib ./xla_extension/lib/libxla_extension.so
curl -s -L $C_XLA_URL | tar xz -C /usr/local/lib --strip-components 1

bash -c 'echo "/usr/local/lib/xla_extension/lib" >> /etc/ld.so.conf.d/xla_extension.conf' && ldconfig
bash -c 'echo "/usr/local/lib/c_xla_extension/lib" >> /etc/ld.so.conf.d/c_xla_extension.conf' && ldconfig
