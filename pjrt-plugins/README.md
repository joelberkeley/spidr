# PJRT Plugins

A PJRT plugin provides the compiler and hardware device support required to execute spidr graphs. We provide plugins for [CPU](xla-cpu/README.md) and [CUDA-enabled GPUs](xla-cuda/README.md). You can also use third-party plugins, or make your own.

## How to integrate your own plugin

Third-party vendors have written a number of PJRT plugins which we haven't integrated into spidr, such as for ROCm-enabled (AMD) GPUs, and TPUs. These plugins all share a common API, written in C, and it requires relatively little code to integrate these into spidr. Each device and compiler may require its own specific configuration and dependencies. We provide Idris APIs to configure some PJRT components, but not all. If you need a configuration option that is not exposed in Idris, or would like any other assistance, [contact us](../README.md#contact).

To integrate a plugin, you need to implement a function that produces an Idris `Device`. The core component is a `PjrtApi`, which is a thin wrapper round a pointer to a C `PJRT_Api` struct. See the documentation for [`device`](https://joelberkeley.github.io/spidr/docs/Device.html#Device.device) for more information.