# XLA PJRT plugin for CUDA-enabled GPUs

This is the PJRT plugin for CUDA-enabled GPUs. It uses the XLA compiler.

## Install

This plugin requires Linux, a CUDA-enabled GPU, and a number of Nvidia packages.

First, install Nvidia GPU drivers. You can either install the remaining packages on your host machine, or use the provided Docker container. The Docker container is more reliable.

### Install with Docker

To install with Nvidia Docker, first install [Docker](https://www.docker.com/), then the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). Next, build the Docker container with, from the repository root
```sh
docker build -t spidr -f pjrt-plugins/xla-cuda/Dockerfile .
```
and run it with e.g.
```sh
docker run -it --gpus all --name spidr spidr bash
```
Next, in the container, install [pack](https://github.com/stefan-hoeck/idris2-pack) and prerequisites (the container uses Ubuntu). Finally, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```

### Install without Docker

_*NOTE*: Tested on Ubuntu 24.04. We cannot guarantee these instructions won't break existing installations of CUDA_.

To install on your host machine, install the CUDA libraries as demonstrated in the [Dockerfile](Dockerfile). Then install the plugin with
```
pack install pjrt-plugin-xla-cuda
```
