FROM ubuntu:24.04

RUN apt-get update && apt-get install -y git curl clang lld build-essential
RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-amd64.deb
RUN apt-get install -y ./bazelisk-amd64.deb
