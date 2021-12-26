# spidr

**Note:** spidr is in early development. Many APIs have no implementation at this time.

With spidr, we explore what is possible when we bring the latest developments in programming language theory and hardware acceleration to probabilistic modelling. We hope to help developers find new ways to write and verify robust, performant and practical machine learning utilities, libraries and frameworks; allow machine learning researchers to leverage software design to find new research avenues with tools that are easy to compose, modify and extend; and allow those new to machine learning to learn about common or useful algorithms. To these ends, we aim to make spidr

  - **robust** by leveraging the dependent and quantitative types and theorem proving offered by [Idris](https://github.com/idris-lang/Idris2), alongside carefully considered testing
  - **performant** by using [XLA](https://www.tensorflow.org/xla) for efficient graph compilation for the GPU, TPU and other hardware
  - **composable** via a purely functional API
  - **practical** with lightweight and intuitive APIs
  - **informative** with clear and extensive documentation

This is a tall order, so to keep the workload manageable we may omit conceptually similar algorithms where they don't contribute new insights in design or machine learning computation. This emphasis on design over completeness is spidr's distinctive feature.

Please use spidr responsibly. We ask that you ensure any benefits you gain from this are used to help, not hurt.

spidr has an [online API reference](https://joelberkeley.github.io/spidr), and [tutorials](tutorials). The tutorials are [literate files](https://idris2.readthedocs.io/en/latest/reference/literate.html) and can be executed like any other source file.

We use [semantic versioning](https://semver.org/).

## Installation

First, obtain a release of spidr from the [releases page](https://github.com/joelberkeley/spidr/releases), and extract the downloaded archive. Next obtain an XLA binary from [elixir-nx/xla](https://github.com/elixir-nx/xla/releases). Extract the archive into the `backend/` directory in the spidr release downloaded in the last step.

[Install Idris 2](https://github.com/idris-lang/Idris2/blob/main/INSTALL.md). We recommend using [Homebrew](https://brew.sh/) if you're unsure which option to use. Finally, in the spidr root directory, install spidr with
```bash
idris2 --install spidr.ipkg
```

## Contact

To ask for new features or to report bugs, make a new GitHub issue. For any other questions or comments, message @joelb on the [Idris community discord](https://discord.gg/YXmWC5yKYM).
