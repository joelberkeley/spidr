# The spidr Compiler

The `Foreign` directory contains the bare foreign functions to the C wrapper around XLA.

The `TensorFlow` directory contains functionality that wraps the foreign functions in `Foreign` such that the type signature express contracts of the XLA API. It is not opinionated, in that it only attempts to provide a reasonably type-safe Idris interface to XLA, and does not attempt to reinterpret the API in a more idiomatic or convenient form.

The `XLA` module wraps the functionality in `TensorFlow` in a convenient form that diverges from the XLA API.
