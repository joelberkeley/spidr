<!--
Copyright 2022 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# The Design and Philosophy of spidr's Tensor

spidr was born out of a desire to explore what is possible when we have the opportunity to use the latest technologies available in engineering machine learning systems. This philosophy has led many central decision in spidr's design. In this tutorial, we'll explore these, detailing spidr's architecture and limitations as we go.

## Dependent types: precisely describing Tensors

The first technology that users will likely notice is that spidr utilises the dependent types offered by Idris. Dependent types allow you to, among other things, include values at type level. Here's an introductory [talk](https://www.youtube.com/watch?v=mOtKD7ml0NU) on dependent types, and a [book](https://www.manning.com/books/type-driven-development-with-idris) if you want a deeper dive. Let's have a quick look at an example of relevance to spidr. A `List Int` is a list of integers. It's not a dependent type, and it can have zero, five or two hundred elements. which can have any number of elements
<!-- idris
import Data.Vect
-->
```idris
xs : List Int
xs = [0, 1, 2]

xs' : List Int
xs' = []
```
If we try to write a function `head` which gives us the first element, we immediately run into a problem
```idris
head : List Int -> Int
head [] = ?hmmm
head (x :: _) = x
```
we can't implement `head` for empty lists, at least not as a total function. Let's turn to another type. A `Vect 3 Int` is also a list, but it's a dependent type which always contains precisely three integers
```idris
ys : Vect 3 Int
ys = [0, 1, 2]

-- this won't compile
ys' : Vect 3 Int
ys' = []
```
If we try to define `head` for `Vect`, we find that we can use the types to require that the argument isn't empty
```idris
head : Vect (S n) Int -> Int
head [] impossible  -- `impossible` means this case can't happen
head (x :: _) = x
```
This kind of precision can be used not only to constrain arguments, but also function return values. For example, consider the nested list, or matrix
```idris
zs : Vect 2 (Vect 3 Int)
zs = [[3, 5, 7],
      [2, 3, 3]]
```
We could define a function that transposes matrices such as this, where the "shape" of the resulting matrix can be written in terms of the input matrix, all at type level
```idris
transpose : Vect m (Vect n Int) -> Vect n (Vect m Int)
```
It's exactly this kind of extra precision that we use throughout spidr when working with tensors.

## Running Tensors with XLA

When writing spidr's `Tensor` API, we had the option to implement `Tensor` within spidr, or use a third party offering. We chose to opt for a third party tool for a number of reasons:

* it allowed us to start working on higher-level aspects of spidr sooner, such as the probabilistic modelling API
* many frameworks have been highly optimized and offer impressive performance
* many frameworks offer acceleration on hardware beyond CPUs

We were drawn to smaller third party tools that offered only what we needed and nothing more, and especially those that were newer as they would be more likely to have learnt from older frameworks. The first candidate was Graphcore's Poplar. We ruled this out because IPUs are difficult to access for individuals, and other accelerators are either not emphaised or not supported at all. XLA was the next candidate. It supports a number of accelerators of interest, and it is currently being used by Google's JAX, which implies it will remain active for the forseeable future. It also offers a C++ API which allows us to efficiently call into it from Idris. In retrospect, progress has been slower due to the fact that XLA does not include automatic differentiation. We're unsure if this would have affected our decisions had we factored this into our decision.

As mentioned, XLA has a C++ API. In order to call this from Idris, we had two options. The first is to write a C++ backend for Idris. Apparently the Idris core language is small, which means writing new backends is less work than one may expect. The other option is to wrap XLA in a pure C wrapper and use Idris' FFI capabilities to call into this wrapper. We chose this second option for a number of reasons:

* we had more familiarity with how to FFI into C than we did in writing a backend
* all Idris backends support C FFI, which means that spidr would be compatible with all Idris backends
* the C wrapper may be useful for other projects as many languages support FFI to C

We decided to keep the C wrapper as close to the C++ interface as possible, within the limitations of what would work with Idris' FFI. We did this so that the C wrapper made few assumptions about how it would be used, and also so that we could implement as much of the logic as possible within Idris, simply because we find Idris easier to work with. Of course, this means there is a layer in Idris of very XLA-specific code that looks only approximately like spidr's tensor API. Thus, spidr is structured as
```
    Idris tensor API ------> XLA-specific Idris layer ------> C wrapper for XLA ------> XLA C++ interface
```
