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

spidr is loosely designed around [StableHLO](https://openxla.org/stablehlo), a set of operations for machine learning programs with strong compatibility guarantees.

> *__DETAIL__* StableHLO is based on [MHLO](https://github.com/tensorflow/mlir-hlo), and uses [MLIR](https://mlir.llvm.org/) bytecode as a serialization format; MLIR is a sub-project of [LLVM](https://llvm.org/).

spidr represents each graph as a topologically-sorted stack of `Expr` values, each of which corresponds (almost) one-to-one with a StableHLO operation. The primary runtime work of spidr is two-fold: build the stack, then interpret it with FFI calls to the StableHLO API. We'll take you through each of these steps in turn.

!!!! what about executing the stablehlo, which is runtime work?

> *__DETAIL__* spidr currently builds XLA rather than StableHLO programs, then converts these into HLO. In future, we will build StableHLO directly. The XLA and StableHLO APIs are almost identical.

## The Idris tensor graph

Each node in our graph encodes a single tensor operation. Let's look at a very simple graph representation and iterate on that towards what we use in spidr. We can represent literals, addition and multiplication by the type
```idris
data Expr =
    Lit Int
  | Add Expr Expr
  | Mul Expr Expr
````
This works, but quickly becomes extremely wasteful, as we can see when we write out the expression z &times; z where z = 7 + 9:
```idris
Mul (Add (Lit 7) (Lit 9)) (Add (Lit 7) (Lit 9))
```
Not only do we store z twice, but we lose the information that it's the same calculation, so we either also compute it twice, or have to inspect the expression to eliminate common subexpressions. For graphs of any reasonable size, this is inadmissible. We solve this by labelling each `Expr` node that appears in our computational graph. These labels are essentially pointers. spidr could ask the user to provide these labels, but opts to generate them itself.
 `Expr` nodes can refer to other nodes via the label, rather than the value itself, and they could do this in one a number of ways. We'll show a couple. In each of these cases, our labels are `Nat`.

The first option is to bake the labelling into the data type itself, like
```idris
data ExprL =
    Lit Int
  | Add Nat Nat
  | Mul Nat Nat
  | Let Nat Expr Expr
```
Notice how the arguments to `Add` and `Mul` are now labels, rather than `Expr` values themselves. Our earlier example becomes
```idris
Let 0 (Lit 7)           -- label `Lit 7` as 0 in what follows
  $ Let 1 (Lit 9)
    $ Let 2 (Add 0 1)   -- 0 and 1 point to `Lit 7` and `Lit 9`
      $ Mul 2 2         -- each 2 points to `Add 0 1`
```
Another option, a natural representation for a directed acyclic graph such as our computational graph, is a topologically-sorted list: `List Expr`. In this setup we implicitly use the list indices as our labels, and append the appropriate `Expr` to this list for each tensor operation. Our earlier example becomes
```
[ Lit 1
, Lit 2
, Add 0 1
, Mul 2 2 
]
```
spidr uses a list, or stack, of ops.

> *__DETAIL__* Instead of replacing the `Expr` arguments to `Expr` data constructors, such as `Add` and `Mul`, with `Nat` labels, we can introduce a constructor `Var Nat` to refer to labelled nodes. This would allow us to, in the same graph, label nodes and reuse them, and not label other nodes and not reuse them.

> *__DETAIL__* Due to limitations in our current handling of scoping in spidr, node labels are not contiguous and cannot therefore be list indices. Instead, we use a `List (Nat, Expr)` where the `Nat` is a label for the `Expr` node.

!!!!!!!!!!! In this para, how to explain why `Graph` is over `List Expr` not just `Nat`.

 In either of these approaches, we need a notion of state to unambiguously label nodes. Idris is a purely functional language, which means effects, including state, are explicit. In spidr, this state is expressed with the `Graph` type constructor, which is essentially a `State` over our topologically-sorted list. Put another way, `Graph` is the _effect_ of adding nodes to a computation graph.

Explicit state introduces a tradeoff between performance and ergonomics. We discuss in the section ....

Now we know how spidr constructs the graph, let's look at how it consumes it.

## Interpreting the graph

...

### The cost of explicit state: boilerplate and monads

Caching ensures that the computation you write will be the computation sent to the graph compiler. Unfortunately this comes with downsides. First, there is extra boilerplate. Most tensor operations accept `Tensor shape dtype` and output `Graph (Tensor shape dtype)`, so when you compose operations, you'll need to handle the `Graph` effect. For example, what might be `abs (max x y)` in another library can be `abs !(max !x !y)` in spidr. One notable exception to this is infix operators, which accept `Graph (Tensor shape dtype)` values. This is to avoid unreadable algebra: you won't need to write `!(!x * !y) + !z`. However, it does mean you will need to wrap any `Tensor shape dtype` values in `pure` to pass it to an infix operator. Let's see an example:
<!-- idris
import Literal
import Tensor
-->
```idris
f : Tensor shape F64 -> Tensor shape F64 -> Graph $ Tensor shape F64
f x y = (abs x + pure y) * pure x
```
Here, `pure` produces a `Graph (Tensor shape F64)` from a `Tensor shape F64`, as does `abs` (the element-wise absolute value function). Addition `(+)` and multiplication `(*)` produce _and accept_ `Graph (Tensor shape F64)` so there is no need to wrap the output of `abs x + pure y` in `pure` before passing it to `(*)`. A rule of thumb is that you only need `pure` if both of these are true

* you're passing a tensor to an infix operator
* the tensor is either a function argument or is on the left hand side of a monadic bind `x <- expression`

Second, care is needed when reusing expressions to make sure you don't recompute sections of the graph. For example, in
```idris
whoops : Graph $ Tensor [3] S32
whoops = let y = tensor [1, 2, 3]
             z = y + y
          in z * z
```
`z` will be calculated twice, and `y` allocated four times (unless the graph compiler chooses to optimize that out). Instead, we can reuse `y` and `z` with
```idris
ok : Graph $ Tensor [3] S32
ok = do y <- tensor [1, 2, 3]
        z <- pure y + pure y
        pure z * pure z
```
Here, `y` and `z` will only be calculated once. This problem can occur more subtley when reusing values from another scope. For example, in
```idris
expensive : Graph $ Tensor [] F64
expensive = reduce @{Sum} [0] !(fill {shape = [100000]} 1.0)

x : Graph $ Tensor [] F64
x = abs !expensive

y : Graph $ Tensor [] F64
y = square !expensive

whoops' : Graph $ Tensor [] F64
whoops' = max !x !y
```
`expensive` is calculated twice. Instead, you could pass the reused part as a function argument
```idris
xf : Tensor [] F64 -> Graph $ Tensor [] F64
xf e = abs e

yf : Tensor [] F64 -> Graph $ Tensor [] F64
yf e = square e

okf : Tensor [] F64 -> Graph $ Tensor [] F64
okf e = max !(xf e) !(yf e)

res : Graph $ Tensor [] F64
res = okf !expensive
```
Note we must pass the `Tensor [] F64` to `xf`, `yf` and `okf`, rather than a `Graph (Tensor [] F64)`, if the tensor is to be reused.
