# XLA PJRT plugin for CUDA-enabled GPUs

This is the PJRT plugin for CUDA-enabled GPUs. It uses the XLA compiler.

## Install

This plugin requires Linux, a CUDA-enabled GPU, and a number of Nvidia packages. We have tested this only on Ubuntu 24.04.

First, install NVIDIA GPU drivers. Next, install CUDA toolkit 12.6, as well as cuDNN and NCCL, making sure the versions are compatible with CUDA 12.6 and your OS. Finally, install the plugin with
```
pack install pjrt-plugin-xla-cuda
```
