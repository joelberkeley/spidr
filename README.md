# spidr

_Accelerated machine learning with dependent types_

spidr is in early development. The core linear algebra API is largely complete, but we are missing automatic differentiation and gradient-based optimizers.

To install, see [the instructions](INSTALL.md). We use [semantic versioning](https://semver.org/). See the online [API reference](https://joelberkeley.github.io/spidr/) (available for the latest release), and the [tutorials](tutorials) for extended discussions of spidr's architecture. The tutorials are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

### Motivation

We made spidr to try out modern programming language capabilities in machine learning systems. To this end, we chose [Idris](https://github.com/idris-lang/Idris2) for the API; Idris is a general-purpose purely functional programming language with a particularly expressive type system. We also wanted to build something performant enough for working machine learning practitioners. Since implementing efficient low-level linear algebra is not one of the project goals, we opted to build on existing compiler and hardware accelerator technologies, and chose [XLA](https://github.com/openxla/xla) as our first backend.

### What can spidr do?

#### Catch errors before runtime

If your spidr program compiles, and your hardware can run it, then it will run. This is primarily because Idris checks tensor shapes during compilation. For example, this will compile
<!-- idris
import Literal
import Tensor
-->
```idris
x : Ref $ Tensor [3] S32
x = tensor [1, 2, 3]

y : Ref $ Tensor [3] S32
y = x + tensor [0, 1, 2]
```
but this won't
```idris
failing "elaboration"
  z : Ref $ Tensor [3] S32
  z = x + tensor [0, 1]
```
because you can't add a vector of length two to a vector of length three. Shape manipulation extends beyond comparing literal dimension sizes to arbitrary symbolic manipulation
```idris
append : Tensor [m, p] F64 -> Tensor [n, p] F64 -> Ref $ Tensor [m + n, p] F64
append x y = concat 0 x y
```
As a bonus, spidr programs are reproducible. Any one graph will always produce the same result when run on the same hardware.

#### Optimized compilation for hardware accelerators

spidr benefits from much of what XLA has to offer, namely the performance benefits of optimizations such as fusion, and execution on various hardware accelerators. spidr programs currently run on CPU and GPU.

#### Graph generation

This is a high-priority feature but is not yet implemented. spidr can generate new tensor graphs from existing ones. We plan to use this to implement vectorization, just-in-time compilation, and automatic differentiation like JAX's [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) and [`grad`](https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html#grad).

### Acknowledgements

I'd like to thank the Idris community for their frequent guidance and Idris itself, the Numerical Elixir team for their XLA binaries, Secondmind colleagues for discussions around machine learning design, friends and family for their support, Google for XLA, and Github for hosting.

### Contact

To ask for new features or to report bugs, make a new GitHub issue. For any other questions or comments, message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
