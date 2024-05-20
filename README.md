# spidr

_Accelerated machine learning with dependent types_

spidr is in early development. The linear algebra API is largely complete, but we are missing automatic differentiation and gradient-based optimizers.

See the [online reference](https://joelberkeley.github.io/spidr/) for API documentation, and the [tutorials](tutorials) for extended discussions of spidr's architecture. The tutorials are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

### Install

spidr comes in two parts: the core library to write computations, and PJRT plugins to run them. The plugins depend on the core library, so you only need to install a plugin. You can either install the CPU plugin, with 
```
pack install pjrt-plugin-xla-cpu
```
or read the [plugin documentation](pjrt-plugins/README.md) for the CUDA-enabled GPU plugin and custom plugin builds.

We have built and tested spidr on Ubuntu Linux 22.04. If you're on another platform and spidr doesn't work for you, [get in touch](#contact). We're keen to support your daily driver.

### Motivation

We made spidr to try out modern programming language capabilities in machine learning systems. To this end, we chose [Idris](https://github.com/idris-lang/Idris2) for the API; Idris is a general-purpose purely functional programming language with a particularly expressive type system. We also wanted to build something performant enough for working machine learning practitioners. Implementing efficient low-level linear algebra is not one of the project goals, so we opted to build on existing compiler and hardware accelerator technologies. OpenXLA's [PJRT](https://openxla.org/) provides straightforward compatibility with many such devices and compilers.

### What can spidr do?

#### Catch errors before runtime

If your spidr program compiles, and your hardware can run it, then it will run. This is primarily because Idris checks tensor shapes during compilation. For example, this will compile
<!-- idris
import Literal
import Tensor
-->
```idris
x : Graph $ Tensor [3] S32
x = tensor [1, 2, 3]

y : Graph $ Tensor [3] S32
y = x + tensor [0, 1, 2]
```
but this won't
```idris
failing "elaboration"
  z : Graph $ Tensor [3] S32
  z = x + tensor [0, 1]
```
because you can't add a vector of length two to a vector of length three. Shape manipulation extends beyond comparing literal dimension sizes to arbitrary symbolic manipulation
```idris
append : Tensor [m, p] F64 -> Tensor [n, p] F64 -> Graph $ Tensor [m + n, p] F64
append x y = concat 0 x y
```
As a bonus, spidr programs are reproducible. Any one graph will always produce the same result when run on the same hardware.

#### Execute on hardware accelerators

You can run spidr programs on any hardware for which there's a PJRT plugin. CPU and CUDA plugins are available out of the box. You can also create your own plugins with minimal code, see [the guide](pjrt-plugins/README.md) for instructions. The tooling required to build a plugin exist for ROCm-enabled GPUs, specialised machine learning accelerators, and more.

#### Optimize graph compilation

Each PJRT plugin contains a graph compiler, and there are multiple such compilers to choose from. The plugins we provide out of the box use the XLA compiler, which offers substantial performance benefits through e.g. [CSE and operator fusion](https://openxla.org/xla/architecture).

#### Graph generation

This is a high-priority feature but is not yet implemented. spidr can generate new tensor graphs from existing ones. We plan to use this to implement vectorization and automatic differentiation like JAX's [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap) and [`grad`](https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html#grad).

### Acknowledgements

I'd like to thank the Idris community for their frequent guidance and Idris itself, the Numerical Elixir team for their early binaries, Secondmind colleagues for discussions around machine learning design, friends and family for their support, Google and OpenXLA for the compiler stack, and Github for hosting.

### Contact

To ask for new features or to report bugs, make a new GitHub issue. For any other questions or comments, message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
