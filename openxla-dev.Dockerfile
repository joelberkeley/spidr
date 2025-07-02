FROM ubuntu:24.04

RUN apt-get update && apt-get install -y git curl build-essential
RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-amd64.deb
RUN dpkg -i bazelisk-amd64.deb && rm bazelisk-amd64.deb
