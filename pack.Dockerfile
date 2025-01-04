FROM ubuntu:24.04

RUN apt-get update && apt-get install -y git libgmp3-dev build-essential chezscheme
RUN git clone https://github.com/stefan-hoeck/idris2-pack.git
RUN cd idris2-pack && make micropack SCHEME=chezscheme
RUN rm -r /idris2-pack
RUN ~/.pack/bin/pack switch HEAD

ENV PATH=/root/.pack/bin:$PATH
