# spidr

**NOTE** spidr is in very early development. It is not ready for use.

With spidr, we explore what is possible when we bring the latest developments in programming language theory and machine learning hardware to probabilistic modelling. We hope to help developers find new ways to write and verify robust, performant and practical machine learning utilities, libraries and frameworks; allow machine learning researchers to leverage software design to find new research avenues with tools that are easy to compose, modify and extend; and allow those new to machine learning to learn about common or useful algorithms. To these ends, we aim to make spidr

  - **robust** by leveraging the dependent and quantitative types and theorem proving offered by [Idris](https://github.com/idris-lang/Idris2), and carefully considered testing
  - **performant** by using [Graphcore's PopLibs library](https://github.com/graphcore/poplibs) for machine learning on the IPU
  - **composable** via a purely functional API
  - **practical** with lightweight and intuitive APIs
  - **informative** with clear and extensive documentation

We may choose to omit conceptually similar algorithms where they don't contribute to insights in design or machine learning computation.

Please use spidr responsibly. We ask that you ensure any benefits you gain from this are used to help, not hurt.

spidr has an [online API reference](https://joelberkeley.github.io/spidr), and tutorials in the [tutorials](tutorials) directory.

We use [semantic versioning](https://semver.org/). Contributions are welcome.
