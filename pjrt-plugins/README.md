# PJRT Plugins

A PJRT plugin provides the compiler and hardware device support required to execute spidr graphs. We already provide support for CPUs and CUDA-enabled GPUs. These plugins both use the XLA compiler.

## How to integrate your own plugin

Third-party vendors have written a number of PJRT plugins which we haven't integrated into spidr, such as for ROCm-enabled (AMD) GPUs, and TPUs. These plugins all share a common API, written in C, and it requires relatively little code to integrate these into spidr. Note it will be up to you to configure the plugin correctly. We provide Idris APIs to configure some PJRT components, but not all. If you need a configuration option that is not exposed in Idris, let us know in a GitHub issue.

To integrate a plugin, you need to implement everything required to create an Idris `Device`. The core component is a `PjrtApi`, which is a thin wrapper round a pointer to a C `PJRT_Api` struct. See the documentation for [`device`](https://joelberkeley.github.io/spidr/docs/Device.html#Device.device) for more information.
