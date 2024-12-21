# XLA PJRT plugin for CUDA-enabled GPUs

This is the PJRT plugin for CUDA-enabled GPUs. It uses the XLA compiler.

## Install

This plugin requires Linux, a CUDA-enabled GPU, and a number of Nvidia packages.

First, install Nvidia GPU drivers. The remaining packages can be installed in two different ways: with Docker, which is reliable but cumbersome; or without Docker, where installing Nvidia packages can prove tricky indeed. We recommend using Docker.

### Install with Docker

To install with Nvidia Docker, first install [Docker](https://www.docker.com/), then the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). Next, run the Nvidia TensorRT Docker container with e.g.
```
docker run -it                          \
    --gpus all                          \
    --name spidr                        \
    -v $(pwd):/spidr                    \
    -w /spidr                           \
    nvcr.io/nvidia/tensorrt:24.07-py3   \
    bash
```
The image version `24.07-py3` is important. Next, in the Docker container, install `pack` and prerequisites (the container uses Ubuntu). Finally, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```

### Install without Docker

To install without Docker, first install CUDA toolkit 12.3. Then install the cuDNN and TensorRT packages. We have successfully installed these last two, on Ubuntu 22.04, with
```
apt-get install libcudnn8 libnvinfer8 libnvinfer-plugin8
```
Finally, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```
