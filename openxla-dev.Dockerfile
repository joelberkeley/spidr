FROM ubuntu:24.04

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update && apt-get install -y git curl python3 clang lld build-essential
RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-amd64.deb
RUN dpkg -i bazelisk-amd64.deb && rm bazelisk-amd64.deb
