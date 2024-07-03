# spidr tests with XLA CUDA plugin

Build the image from the repository root with
```bash
$ docker build -t xla-cuda -f test/xla-cuda/Dockerfile .
```
then run tests with
```bash
$ docker run --name xla-cuda --gpus all xla-cuda
```
