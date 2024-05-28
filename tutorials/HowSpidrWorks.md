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
# How spidr works

In this tutorial, we explain how spidr runs the tensor code you write in Idris.

## StableHLO: the tensor graph representation

spidr is loosely designed around [StableHLO](https://openxla.org/stablehlo), a set of operations for machine learning programs offering portability with several compilers, along with version compatibility guarantees.

> *__DETAIL__* StableHLO is implemented as an [MLIR](https://mlir.llvm.org/) dialect, and is based on another (deprecated) dialect MLIR-HLO. Such dialects are typically "lowered" down to another dialect compiled by the [LLVM](https://llvm.org/) compiler. However, no compilers currently take this route for StableHLO. XLA, for example, simply converts StableHLO to HLO, its own internal graph representation, which is not an MLIR dialect.

spidr represents each graph as a topologically-sorted stack of `Expr` values, each of which corresponds (almost) one-to-one with a XLA tensor operation. Most of these ops are also present in the StableHLO specification. spidr uses the XLA API to build an HLO program. The primary runtime work of spidr is three-fold: build the stack; interpret it as an HLO program; compile and execute the HLO. We'll take you through each of these steps in turn.

!!!! what about executing the stablehlo, which is runtime work?

## Building the tensor graph in Idris

Each node in our graph encodes a single tensor operation. Let's look at a very simple graph representation and iterate on that towards what we use in spidr. We can represent literals, addition and multiplication by the type
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
Not only do we store z twice, but we lose the information that it's the same calculation, so we either also compute it twice, or have to inspect the expression to eliminate common subexpressions. For graphs of any reasonable size, this is inadmissible. We solve this by labelling each `Expr` node that appears in our computational graph. These labels are essentially pointers. spidr could ask the user to provide these labels, but opts to generate them itself.
 `Expr` nodes can refer to other nodes via the label, rather than the value itself, and they could do this in one a number of ways. We'll show a couple. In each of these cases, our labels are `Nat`.

The first option is to bake the labelling into the data type itself, as
```idris
data Expr
  = Lit Int
  | Add Nat Nat
  | Mul Nat Nat
  | Let Nat Expr Expr
```
Notice how the arguments to `Add` and `Mul` are now labels, rather than `Expr` values. Our earlier example becomes
```idris
Let 0 (Lit 7)           -- label `Lit 7` as 0 in what follows
  $ Let 1 (Lit 9)
    $ Let 2 (Add 0 1)   -- 0 and 1 point to `Lit 7` and `Lit 9`
      $ Mul 2 2         -- each 2 points to `Add 0 1`
```
Another option, a natural representation for a directed acyclic graph such as our computational graph, is a topologically-sorted list, `List Expr` for
```idris
data Expr
  = Lit Int
  | Add Nat Nat
  | Mul Nat Nat
```
In this setup we implicitly use the list indices as our labels, and for each tensor operation, append the appropriate `Expr` to this list. Our earlier example becomes
```
[ Lit 7
, Lit 9
, Add 0 1
, Mul 2 2 
]
```
spidr uses this second approach of a list, or stack, of `Expr`s.

> *__DETAIL__* Instead of replacing the `Expr` arguments to `Expr` data constructors, such as `Add` and `Mul`, with `Nat` labels, we can introduce a constructor `Var Nat` to refer to labelled nodes. This would allow us to only label a node if we plan on reusing it. We don't currently offer this in spidr.

> *__DETAIL__* Due to limitations in our current handling of scopes in spidr, node labels are not contiguous and cannot therefore be list indices. Instead, we use a `List (Nat, Expr)` where the `Nat` is a label for the `Expr` node.

!!!!!!!!!!! In this para, how to explain why `Graph` is over `List Expr` not just `Nat`.

 In either of these approaches, we need a notion of state to unambiguously label nodes. Idris is a purely functional language, which means effects, including state, are explicit. In spidr, this state is expressed with the `Graph` type constructor, which is essentially a `State` over our topologically-sorted list. Put another way, `Graph` is the _effect_ of adding nodes to a computation graph.

Explicit state introduces a tradeoff between performance and ergonomics. We discuss this in the tutorial [Nuisances in the tensor API](Nuisances.md).

### Graph manipulation

...

Now we know how spidr constructs the graph, let's look at how it consumes it.

## Interpreting the graph with XLA

spidr next converts the stack of tensor operations from its own internal representation to HLO. The process is fairly straightforward, and involves creating a fixed-length `IOArray` cache of pointers to C++ XLA `XlaOp`s, then iterating over the stack, creating an `XlaOp` for each `Expr`. Unlike a `List`, the `IOArray` provides O(1) access to previously-created `XlaOp`s for when we need to fetch by label. The process requires heavy use of the Idris C FFI and makes use of a thin custom C wrapper round the XLA C++ API.

## Compiling and executing the graph with PJRT

...
