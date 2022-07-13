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
# Dependently-Typed Tensors

## What are dependent types?

spidr utilises the dependent types offered by Idris. Dependent types allow you to have values in types (here's an introductory [talk](https://www.youtube.com/watch?v=mOtKD7ml0NU) on dependent types, a [tutorial](https://github.com/stefan-hoeck/idris2-tutorial) and a [book](https://www.manning.com/books/type-driven-development-with-idris)). Let's see an example of relevance to spidr. A `List Int` is a list of integers. These are representable in any language from C++ to Swift. It's not a dependent type, and it can have any number of elements
<!-- idris
import Data.Vect
-->
```idris
xs : List Int
xs = [0, 1, 2]

xs' : List Int
xs' = []
```
It works great for operations that are blind to the number of elements, like iteration and sorting, but if we want to access specific elements, we come across difficulties. Let's write a function `head` which gives us the first element of a list:
```idris
head : List Int -> Int
head [] = ?hmmm
head (x :: _) = x
```
We have a problem. `head` requires there is an initial element to return, which empty lists don't have. Put another way, we don't have any evidence that the list has an element to return. Dependent types allow us to provide this evidence. Let's see an example. A `Vect n Int` is also a list of integers, but unlike `List Int` it's a dependent type which always contains precisely `n` integers (where `n` is a natural number). Here's an example:
```idris
ys : Vect 3 Int
ys = [0, 1, 2]
```
If we tried to implement this with `ys = [0]`, it wouldn't compile, as `[0]` is a `Vect 1 Nat`. With the extra information of how many elements are in the list, we can now define `head` for `Vect`:
```idris
namespace Vect
  head : Vect (S n) Int -> Int
  head [] impossible
  head (x :: _) = x
```
Here, `impossible` indicates that this particular case can never happen, because the types provide evidence that the number of elements is never zero (`S n` means any number one greater than another). This kind of precision can be used not only to constrain arguments, but also function return values. For example, consider the nested list, or matrix
```idris
zs : Vect 2 (Vect 3 Int)
zs = [[3, 5, 7],
      [2, 3, 3]]
```
We could define a function that transposes matrices such as this, where the shape of the resulting matrix can be written in terms of the input matrix, all at type level
```idris
transpose : Vect m (Vect n Int) -> Vect n (Vect m Int)
```
`transpose zs` will give us a `Vect 3 (Vect 2 Int)`. It's precisely this kind of extra precision that we use throughout spidr when working with tensors.
