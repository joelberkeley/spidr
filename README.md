# spidr

**Note:** spidr is in early development. [See here](https://github.com/joelberkeley/spidr/labels/notable%20limitation) for a list of notable limitations.

With spidr, we explore what is possible when we bring some of the latest developments in programming language theory and hardware acceleration to probabilistic modelling. We hope to help developers find new ways to write and verify robust, performant and practical machine learning utilities, libraries and frameworks; allow machine learning researchers to leverage software design to find new research avenues with tools that are easy to compose, modify and extend; and allow those new to machine learning to learn about common or useful algorithms. To these ends, we aim to make spidr

  - **robust** by leveraging the dependent types and theorem proving offered by [Idris](https://github.com/idris-lang/Idris2)
  - **performant** by using [XLA](https://www.tensorflow.org/xla) for efficient graph compilation for the GPU, TPU and other hardware
  - **composable** via a purely functional API
  - **practical** with lightweight and intuitive APIs
  - **informative** with clear and extensive documentation

This is a tall order, so to keep the workload manageable we may omit conceptually similar algorithms where they don't contribute new insights in design or machine learning computation. This emphasis on design over completeness is spidr's distinctive feature.

Please use spidr responsibly. We ask that you ensure any benefits you gain from this are used to help, not hurt.

spidr has an _incomplete_ [online API reference](https://joelberkeley.github.io/spidr), incomplete as the documentation builder is itself a work in progress: it both omits items, and renders items incorrectly. See the source code for full details). spidr also has [tutorials](tutorials), which are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

We use [semantic versioning](https://semver.org/).

I'd like to thank the Idris community for their frequent guidance and Idris itself, the Numerical Elixir team for their XLA binaries, Secondmind colleagues for discussions around machine learning design, friends and family for their support, Google for XLA, and Github for hosting. There are many more I've not mentioned.

## Installation

1. Clone or download the spidr source code at tag v0.0.4 with, for example,
   ```bash
   git clone --depth 1 --branch v0.0.4 https://github.com/joelberkeley/spidr.git
   ```
   See the git history for installing earlier versions.
2. [Install Idris 2](https://github.com/idris-lang/Idris2/blob/main/INSTALL.md). We recommend using [Homebrew](https://brew.sh/) if you're unsure which option to use.
3. In the spidr root directory, install with
   ```bash
   idris2 --install spidr.ipkg
   ```
4. When building your Idris code that depends on spidr, either include `-p spidr` on the command line, or list spidr as a dependency in your project's configuration.
5. Download the XLA C interface from the [releases page](https://github.com/joelberkeley/spidr/releases), and extract the archive. The extracted directory can be placed anywhere you wish.
6. Download an XLA binary from [elixir-nx/xla](https://github.com/elixir-nx/xla/releases), and extract the archive. See the spidr releases page for details of which versions are supported. Place the directory `xla_extension` into the same path as the directory `c_xla_extension` extracted in step 5.
7. When running code that depends on spidr, you may need to set `LD_LIBRARY_PATH` to include the location of the libc_xla_extension.so shared library located in `c_xla_extension` extracted in step 5.

## Contact

To ask for new features or to report bugs, make a new GitHub issue. For any other questions or comments, message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
