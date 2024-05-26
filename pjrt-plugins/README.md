# PJRT Plugins

A PJRT plugin provides the compiler and hardware device support required to execute spidr graphs. We provide plugins for [CPU](xla-cpu/README.md) and [CUDA-enabled GPUs](xla-cuda/README.md). You can also use third-party plugins, or make your own.

## How to integrate your own plugin

Third-party vendors have written a number of PJRT plugins that we haven't integrated into spidr, such as for ROCm-enabled AMD GPUs, and TPUs. When we say "PJRT plugin", we mean both the third-party libraries and their Idris wrappers, but try to make it clear from the context which we mean. The third-party plugins all share a common API, written in C, and it requires relatively little code to integrate these into spidr. If you would like any assistance with plugin development, please [contact us](../README.md#contact).

To integrate a plugin, have a read of the [CPU](xla-cpu) and [CUDA](xla-cuda) plugin implementations, which provide idiomatic examples. In summary, you will need to implement an Idris `Device`. A `Device` is made of a `PjrtApi` and a `PjrtClient`. A `PjrtApi` is a thin wrapper round a pointer to a C `PJRT_Api` struct. Each plugin, as exposed in C, typically contains a function that produces a `PJRT_Api`. You can construct a `PjrtClient` from a `PjrtApi` with `pjrtClientCreate`.
