# spidr

**Note:** spidr is in early development. See [here](https://github.com/joelberkeley/spidr/labels/notable%20limitation) for a list of notable limitations.

A linear algebra and probabilistic modelling library supporting dependent types and hardware acceleration. spidr has two array types: `Tensor` is backed by XLA and uses dependent types to guarantee its shape; `Literal` is pure Idris and allows full use of dependent types.

For installation, see [the instructions](INSTALL.md). We use [semantic versioning](https://semver.org/). See the source code for comprehensive API documentation (`src/` excluding `src/Compiler/`), and the [tutorials](tutorials) for extended discussions of spidr's architecture. The tutorials are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

## Motivation

spidr is a research project into engineering in the machine learning space, in which we bring some of the latest developments in programming language theory to linear algebra and probabilistic modelling. We hope to help developers find new ways to write and verify robust, performant and practical machine learning alogrithms; allow machine learning researchers to leverage software design to find new research avenues with tools that are easy to compose, modify and extend; and allow those new to machine learning to learn about common or useful algorithms. To these ends, we aim to make spidr

  - **informative** with clear and extensive documentation
  - **practical** with lightweight and intuitive APIs
  - **composable** with purely functional APIs
  - **performant** by using [XLA](https://www.tensorflow.org/xla) for efficient graph compilation for the GPU, TPU and other hardware
  - **robust** by leveraging the dependent types and theorem proving offered by [Idris](https://github.com/idris-lang/Idris2)

## Acknowledgements

I'd like to thank the Idris community for their frequent guidance and Idris itself, the Numerical Elixir team for their XLA binaries, Secondmind colleagues for discussions around machine learning design, friends and family for their support, Google for XLA, and Github for hosting. There are many more I've not mentioned.

## Contact

To ask for new features or to report bugs, make a new GitHub issue. For any other questions or comments, message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
