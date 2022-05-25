# The spidr Compiler

The `Foreign` directory contains the bare foreign functions to the C wrapper around XLA.

Functionality in the `TensorFlow` directory wraps the foreign functions in `Foreign` to improve the type safety of the interface with XLA by expressing contracts of the XLA API. It tries to be unopinionated, in that it wraps the functionality as-is, but with extra type information.

Functionality in the `XLA` module wraps that in `TensorFlow` in a more convenient, safer, or more idiomatic form. It may diverge from the XLA API.
