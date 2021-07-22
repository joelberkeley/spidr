# spidr

**NOTE** spidr is in very early development. It is not ready for use.

With spidr, we explore what is possible when we bring the latest developments in programming language theory and hardware acceleration to probabilistic modelling. We hope to help developers find new ways to write and verify robust, performant and practical machine learning utilities, libraries and frameworks; allow machine learning researchers to leverage software design to find new research avenues with tools that are easy to compose, modify and extend; and allow those new to machine learning to learn about common or useful algorithms. To these ends, we aim to make spidr

  - **robust** by leveraging the dependent and quantitative types and theorem proving offered by [Idris](https://github.com/idris-lang/Idris2), and carefully considered testing
  - **performant** by using [XLA](https://www.tensorflow.org/xla) for efficient graph compilation for the GPU, TPU and other hardware
  - **composable** via a purely functional API
  - **practical** with lightweight and intuitive APIs
  - **informative** with clear and extensive documentation

This is a tall order, so to keep the workload manageable we may choose to omit conceptually similar algorithms where they don't contribute new insights in design or machine learning computation. This emphasis on design over completeness stands spidr apart from most other machine learning framworks.

Please use spidr responsibly. We ask that you ensure any benefits you gain from this are used to help, not hurt.

spidr has an [online API reference](https://joelberkeley.github.io/spidr), and [tutorials](tutorials). The tutorials are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

We use [semantic versioning](https://semver.org/). Contributions are welcome.

## Installation

**Note:** spidr is not executable at this time. Only type checking is possible.

Install [Idris 2](https://github.com/edwinb/Idris2/blob/main/INSTALL.md). We recommend using [Homebrew](https://brew.sh/) if you're unsure which option to use.

Clone this repository, then install spidr with
```bash
idris2 --install spidr.ipkg
```

You can then `import Tensor` etc. in your `Foo.idr` file and run it in a REPL with
```bash
idris2 -p spidr Foo.idr
```
See `idris2 --help` for more build options.

## Contact

To ask for new features or report bugs, make a new GitHub issue. For any other questions, you can private message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
