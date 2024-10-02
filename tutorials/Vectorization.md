<!--
Copyright 2024 Joel Berkeley

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




### Either move this to _How spidr works_, or modify that notebook to account for vectorization being separate.




# Vectorization

In this tutorial, we explain how spidr implements vectorization.

_This file is not executable._

## What is vectorization?

spidr's tensor API puts restrictions on tensor arguments to operations such as
<!-- idris
import Tensor

%hide Tensor.triangle
%hide Tensor.vmap
-->
```
triangle : Primitive dtype => Triangle -> Tensor [n, n] dtype -> Tensor [n, n] dtype
```
This means you cannot apply `triangle Lower` directly to a tensor such as
```
x : Tensor [2, 2, 2] S32
x = tensor [[[1, 1],
             [1, 1]],
             [2, 2],
             [2, 2]]]
```
to get the lower triangle of each nested tensor:
```
y : Tensor [2, 2, 2] S32
y = tensor [[[1, 0],
             [1, 1]],
             [2, 0],
             [2, 2]]]
```
much as you might expect to in, say NumPy. You could of course explicitly split `x` into its component two-by-two square matrices, apply `triangle` to each of these, and then patch them back together, but this kind of _vectorization_ can be handled far more efficiently by the compiler, using hardware specifics. spidr offers a function
```
vmap : Primitive a =>
       (Tensor sa da -> Tag $ Tensor sb db) ->
       Tensor (n :: sa) da -> Tag $ Tensor (n :: sb) db
```
to promote, or _vectorize_ functions like `triangle Lower`, so that you can map them over one or many leading dimensions, as `vmap (pure . triangle Lower) x`, which yields `y`. There are overloads of `vmap` to vectorize multiple arguments of a function together.

## How does spidr implement vectorization?

In short, `vmap` traces the function it is passed, and creates a new graph representation for a function that would apply to a tensor of the specified shape.

We summarise the spidr internal graph representation in [_How spidr Works_](HowSpidrWorks.md). `StableHLO` and XLA operations allow more general shapes than we expose in their direct counterparts in the Idris API. `Expr`, which internally encapsulates all tensor operations, is the same. Because of this, we can simplify evaluated the modify the graph to work on 
