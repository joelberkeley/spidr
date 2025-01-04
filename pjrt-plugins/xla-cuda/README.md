# XLA PJRT plugin for CUDA-enabled GPUs

This is the PJRT plugin for CUDA-enabled GPUs. It uses the XLA compiler.

## Install

This plugin requires Ubuntu Linux 24.04 (or NVIDIA Docker), a CUDA-enabled GPU, and a number of NVIDIA packages.

First, install NVIDIA GPU drivers. The remaining packages can be installed either on your host machine, or in Docker.

### Install without Docker

To install without Docker, refer to the commands in the [Dockerfile](./Dockerfile). Next, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```

### Install with Docker

To install with NVIDIA Docker, first install [Docker](https://www.docker.com/), then the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). Next, clone this repository, and build the runtime Docker container with
```
docker build -t spidr -f pjrt-plugins/xla-cuda/Dockerfile .
```
Next, run the container with
```
docker run -it --gpus all --name spidr -v <absolute_path>:/spidr -w /spidr spidr bash
```
Where `<absolute_path>` contains any executables and/or source code, present on the host, you wish to access from the container. Next, install `pack` and prerequisites in the container. Finally, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```
Note: If you are also running Ubuntu 24.04 on your host machine, you can instead install `pjrt-plugin-xla-cuda`, and build your code, on the host, then mount and run the executable in the container.
