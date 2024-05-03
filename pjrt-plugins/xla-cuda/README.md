# XLA PJRT plugin for CUDA-enabled GPUs

## Dependencies

This plugin requires Linux, a CUDA-enabled GPU, and a number of Nvidia packages. First, you will need Nvidia GPU drivers. The remaining packages can be installed in two different ways. We recommend an Nvidia Docker container as installing Nvidia packages on your system can become very complicated indeed.

### Install with Nvidia Docker

To install with Nvidia Docker, first install [Docker](https://www.docker.com/), then the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). Finally run your spidr programs in an Nvidia TensorRT Docker container with e.g.
```
$ docker run --rm -v $(pwd):/spidr -w /spidr nvcr.io/nvidia/tensorrt:23.11-py3
```
Note the image version `23.11`.

### Install without Docker

To install without Docker, first install CUDA toolkit 12.3. Then install the cuDNN and TensorRT packages. We have successfully installed these last two with the following command on Ubuntu 22.04
```
$ apt-get install libcudnn8 libnvinfer8 libnvinfer-plugin8
```
