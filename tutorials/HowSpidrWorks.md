<!--
Copyright 2023 Joel Berkeley

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
# How spidr Works

In this tutorial, we explain how spidr runs the tensor code you write in Idris.

_This file is not executable._

## StableHLO: the tensor graph representation

spidr is loosely designed around [StableHLO](https://openxla.org/stablehlo), a set of operations for machine learning programs offering portability with several compilers, along with version compatibility guarantees.

> *__DETAIL__* StableHLO is implemented as an [MLIR](https://mlir.llvm.org/) dialect, and is based on another (deprecated) dialect MLIR-HLO. Such dialects are typically "lowered" down to another dialect compiled by the [LLVM](https://llvm.org/) compiler. However, not all compilers take this route for StableHLO. XLA, for example, converts StableHLO to HLO, its own internal graph representation, which is not an MLIR dialect.

spidr represents every computation as a directed acyclic graph of `Expr` values, each of which corresponds (almost) one-to-one with an XLA HLO tensor operation. Most of these ops are also present in the StableHLO specification. spidr uses the XLA API to build an HLO program. The primary runtime work of spidr is three-fold: build the graph of `Expr`s; interpret it as an HLO program; compile and execute the HLO. We'll take you through each of these steps in turn.

## Building the tensor graph in Idris

Each node in our graph encodes a single tensor operation. Let's look at a toy graph representation and iterate on that towards what we use in spidr. We can represent literals, addition and multiplication by the type
```idris
data Expr
  = Lit Int
  | Add Expr Expr
  | Mul Expr Expr
````
This works, but quickly becomes extremely wasteful, as we can see when we write out the expression z &times; z where z = 7 + 9:
```idris
Mul (Add (Lit 7) (Lit 9)) (Add (Lit 7) (Lit 9))
```
Not only do we store z twice, but we lose the information that it's the same calculation, so we either also compute it twice, or have to inspect the expression to eliminate common subexpressions. For graphs of any reasonable size, this is inadmissible. To solve this, we can tag `Expr` nodes, and refer to these tagged nodes by tag instead of value. spidr could ask the user to provide these tags, but opts to generate them itself. We could implement this in one of a number of ways, we'll show a few. In each of these cases, our tags are `Nat`.

We'll start with how to refer to other nodes by tag, and come back to how we'll actually tag them shortly. One option is to replace constructor arguments with `Nat`, like
```idris
data Expr
  = Lit Int
  | Add Nat Nat
  | Mul Nat Nat
```
but then we'd _always_ need to refer to nodes by tag, and as we explain in [Nuisances in the tensor API](Nuisances.md), tagging expressions introduces a tradeoff between performance and ergonomics we'd rather avoid. It would be better if we could either use nodes directly, _or_ by tag. We can do this by adding a constructor `Var` to our `Expr`, as
```idris
data Expr
  = Lit Int
  | Var Nat
  | Add Expr Expr
  | Mul Expr Expr
```
whose sole purpose is to reference other nodes by tag.

Let's now see how to tag nodes. One option is to bake the tagging into `Expr` itself, via a new constructor `Let`:
```idris
data Expr
  = Lit Int
  | Var Nat
  | Let Nat Expr Expr
  | Add Expr Expr
  | Mul Expr Expr
```
Here, our earlier example becomes
```idris
Let 0 (Add (Lit 7) (Lit 9))  -- name `7 + 9` as `0`
  $ Mul (Var 0) (Var 0)      -- each `Var 0` points to `7 + 9`
```
Another option, which makes use of a common representation for directed acyclic graphs such as our computational graph, is to supplement an expression with a topologically-sorted `List Expr`, of all the tagged nodes. Here, we implicitly use the list indices as our tags: to tag an `Expr`, we simply append it to the list. Our earlier example becomes the expression
```idris
Mul (Var 0) (Var 0)
```
along with the list
```idris
[Add (Lit 7) (Lit 9)]
```
spidr uses this second approach of a supplementary list, or stack, of `Expr`s.

In either of these approaches, we need to keep track of generated tags, so we don't reuse them. Idris is a purely functional language, which means effects, including state, are explicit. In spidr, this state is expressed with the `Tag` type constructor, which is essentially a `State` over our topologically-sorted list. Put another way, `Tag` is the _effect_ of tagging nodes in our computational graph. This technique is called [_observable sharing_](https://www.cse.chalmers.se/~dave/papers/observable-sharing.pdf), and introduces the tradeoff between performance and ergonomics we mentioned earlier.

So far, we've assumed a single scope. However, there are higher-order functions in StableHLO, such as [`sort`](https://openxla.org/stablehlo/spec#sort), [`reduce`](https://openxla.org/stablehlo/spec#reduce), and [`if`](https://openxla.org/stablehlo/spec#if). These functions themselves accept functions, which introduce their own scope, or sub-graphs, that must be constructed before we can construct the complete StableHLO graph. Let's see an example, by implementing `if`. We'll first need to add support for boolean types, for the predicate. Then add an `If` constructor to represent the operation, which uses a `Function` type that encapsulates the nodes tagged in this local scope, and the function result
```idris
data U = I Int | B Bool

mutual
  record Function where
    constructor F
    locals : List Expr
    result : Expr

  data Expr
    = Lit U
    | Var Nat
    | Add Expr Expr
    | Mul Expr Expr
    | If Expr Function Function
```
Let's now see how we'd use it. Say we want to evaluate z &times; z where z = 5 if a predicate is true, and 1 + 2 if it's false. If indeed our predicate is false, this is
```idris
If (Lit $ B False)
   (F [Lit $ I 5] (Mul (Var 0) (Var 0)))
   (F []          (Add (Lit $ I 1) (Lit $ I 2)))
```
accompanied by an empty set `[]` of locals in the scope of `If`.

> *__DETAIL__* spidr's interpreter stores nodes across all program scopes in a single array. The interpreter needs to quickly recall these nodes by tag when it lowers the graph, so we choose indices in this global array as our tags. In contrast, the high-level Idris graph defines scopes separately, using local namespaces as outlined above. Because of this, tags in all but one local namespace do not start at zero, and cannot therefore be list indices. Instead, we use a `List (Nat, Expr)` where the `Nat` is the tag. The lists remain both individually, and collectively, topologically sorted.

Finally, a `Tensor` is simply a wrapper round an `Expr`, a runtime-available `Shape`, and an erased element type. Experiment with `show` on your tensor graphs to see all this in action. Note that we have simplified a number of details, so the representation will look different from above.

Now we know how spidr constructs the graph, let's look at how it consumes it.

## Interpreting the graph with XLA

spidr next converts the graph from its own internal representation to HLO. The process is fairly straightforward. We iterate over the stack, interpret each `Expr` as a C++ `XlaOp`, and add the `XlaOp` pointer to a fixed-length `IOArray` array. The remaining `Expr` not in the stack, we simply interpret as the complete expression defined in terms of `XlaOp`s in the array. Unlike a `List`, the `IOArray` provides constant-time access, so we can cheaply access previously-created `XlaOp`s by tag. The process makes heavy use of the Idris C FFI and a thin custom C wrapper round the XLA C++ API.

In future, we plan instead to build StableHLO rather than XLA HLO programs. In that case, for each `Expr`, we'll create a StableHLO `tensor` instead of an `XlaOp`.

## Compiling and executing the graph with PJRT

The OpenXLA project provides [PJRT](https://openxla.org/xla/pjrt_integration), an abstract interface written in C, for _plugins_ that compile and execute StableHLO (and in some cases HLO) programs for a specific hardware device. Compilers include XLA and [IREE](https://iree.dev/). Devices include CPU, GPU (CUDA, ROCm, Intel), and TPU. A machine learning frontend that produces StableHLO programs can use any PJRT plugin to run these programs.

The setup in spidr is fairly straightforward, and like the previous step, mostly involves C FFI calls. Plugins are typically distributed as shared libraries.
