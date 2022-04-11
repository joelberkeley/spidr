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
# A Dependent-Typed Tensor

## What are dependent types?

The first technology that users will likely notice is that spidr utilises the dependent types offered by Idris. Dependent types allow you to, among other things, include values at type level. Here's an introductory [talk](https://www.youtube.com/watch?v=mOtKD7ml0NU) on dependent types, and a [book](https://www.manning.com/books/type-driven-development-with-idris) if you want a deeper dive. Let's have a quick look at an example of relevance to spidr. A `List Int` is a list of integers. It's not a dependent type, and it can have any number of elements
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
we can't implement `head` for empty lists, at least not as a total function. Let's turn to another type that will help us address this. A `Vect 3 Int` is also a list, but it's a dependent type which always contains precisely three integers
```idris
ys : Vect 3 Int
ys = [0, 1, 2]
```
If we tried to implement this with `ys = [0]`, it wouldn't compile, as `[0]` is a `Vect 1 Nat`. If we try to define `head` for `Vect`, we find that we can use the types to require that the argument isn't empty
```idris
namespace Vect
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

## Ergonomic dependent types

There is a skill in using dependent types. Certain approaches can produce code that compiles, but is unergonomic. In this section, we'll look at a few approaches which help the compiler confirm your code is correct, as well as a few typing dead ends.

First off, is a directionality to type inference introduced by the simple fact that functions aren't, in most cases, invertible.
