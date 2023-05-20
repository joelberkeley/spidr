FROM ghcr.io/stefan-hoeck/idris2-pack:latest

RUN apt update && apt install -y curl

WORKDIR spidr

COPY . .

RUN curl -s -L https://github.com/elixir-nx/xla/releases/download/v$XLA_EXT_VERSION/xla_extension-x86_64-linux-$(cat backend/VERSION).tar.gz | tar xz -C /usr/local/lib ./xla_extension/lib/libxla_extension.so
RUN curl -s -L https://github.com/joelberkeley/spidr/releases/download/c-xla-v$(cat backend/XLA_EXT_VERSION)/c_xla_extension-x86_64-linux-$(cat $VERSION).tar.gz | tar xz -C /usr/local/lib ./c_xla_extension/lib/libc_xla_extension.so
RUN echo "/usr/local/lib/xla_extension/lib" | tee -a /etc/ld.so.conf.d/xla_extension.conf
RUN echo "/usr/local/lib/c_xla_extension/lib" | tee -a /etc/ld.so.conf.d/c_xla_extension.conf
RUN ldconfig

RUN pack install hedgehog

CMD ["pack", "--no-prompt", "build", "test.ipkg"]
