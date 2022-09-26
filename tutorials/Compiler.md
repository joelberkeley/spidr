<!--
Copyright 2021 Joel Berkeley

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

# Compiling the Graph

In this tutorial, I will describe how spidr builds a computational graph from user tensor code. I will start with a caveat that I'm not an expert in programming language theory or compiler design. I'm not attempting to recommend any approaches, but simply to show the problems faced in designing spidr's internals and how we solved them.

## What do we want spidr to do?

As a user, we want to be able to write tensor code such as
```idris
x : Tensor [] F64
x = sin 0.0
```
or
```idris
y : Tensor [] S32
y = let z = 1 + 2 in z * z
```
and we want spidr to build this into an efficient computational graph.

In the future, we'll also want to be able to manipulate existing computational graphs. For example, we may want to vectorize a function, as JAX's `vmap` does:
```idris
f : Tensor [n, p, p] dtype -> Tensor [n, p] dtype
f = vmap diag
```
or differentiate a function with respect to its inputs
```
g : Tensor [n] F64 -> Tensor [] F64

g' : Tensor [n, n] F64 -> Tensor [] F64
g' = grad g
```

THOUGHT: We only need to manipulate functions, not tensors.

## How can we achieve it?

We could calculate tensors by recursively evaluating terms starting at the definition of the tensor of interest. For example, in
```
y : Tensor [] S32
y = let x = 1 + 2 in x * x
```
we see `y = x * x`, so we first evaluate one argument of `(*)`, which is `x`, or `1 + 2`. Then we evaluate the other, which is also `1 + 2`, then multiply them. Whilst correct, this is clearly wasteful as we're calculating `1 + 2` twice. We need to remember the result of expressions that have already been evaluated and reuse them. To do this, we need two things: a way to refer to individual terms, and a place to store what those terms are. In the above, we refer to `x` by its name, but clearly this name isn't available to spidr so we'll need a different approach. Meanwhile, `1 + 2` is what `x` refers to.

The only place we can store that information: the name of each tensor, and its value, is in the `Tensor` itself. So how do we name each tensor? Let's take a look at the graph of the example we had above
```
    1   2
     \ /
      +
     / \
     \ /
      *
```
It's pretty straightfoward to name these uniquely, for example as
```
a-> 1   2 <-b
     \ /
  c-> +
     / \
     \ /
  d-> *
```
but how do we do this programmatically? When we name the `2`, we'd need to know the `1` exists else we'd name them the same thing. When we look at the graph as it is shown above, we can jump between nodes and edges with our eyes. The computer can't do this. Instead, it only knows there's a `1` _and_ a `2` when we try to add them. So what we can do is name the `1` and the `2` initially, then adjust the names when we become aware of a conflict. For those familiar with _Git_, this isn't too dissimilar to resolution of conflicting branches. We start by labelling `1` and `2` both with "a", then when we add them, we rename one of them (it doesn't matter which) to "b".
