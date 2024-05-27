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

spidr is loosely designed around [StableHLO](https://openxla.org/stablehlo), a versioned set of operations for machine learning programs, based on [MHLO](https://github.com/tensorflow/mlir-hlo). StableHLO uses [MLIR](https://mlir.llvm.org/) bytecode as a serialization format; MLIR is a sub-project of [LLVM](https://llvm.org/).

spidr represents each graph as a topologically-sorted stack of `Expr` values, each of which corresponds (almost) one-to-one with a StableHLO operation. The primary runtime work of spidr is two-fold: to build this stack, then interpret it with FFI calls to the StableHLO API. We'll take you through each of these steps in turn.

> *__DETAIL__* spidr currently builds XLA rather than StableHLO programs, then converts these into HLO. In future, we will build StableHLO directly. This detail is not too important as the XLA and StableHLO APIs are almost identical.

## Building the Idris tensor graph

We must label nodes in our computational graph, so that we can reuse them. Else, we will suffer potentially disastrous performance consequences as we recalculate even large parts of the graph, or waste cycles eliminating common expressions. spidr could ask the user to provide these labels, or it could generate them itself. We do the latter. This requires a notion of state, to ensure labels are not ambiguous.

Since a graph takes a natural representation as a topologically-sorted list, we can use the indices of this list as our labels, and simply prepend the appropriate `Expr` to this list each time we perform a tensor operation. Our graph can thus simply be a `List Expr`. It might help to visualise this. The mathematical expression z &times; z where z = 1 + 2 is then written as
```
[ Lit 1
, Lit 2
, Add 0 1  -- 0 and 1 point to `Lit 1` and `Lit 2`
, Mul 2 2  -- both 2s points to `Add 0 1` 
]
```

> *__DETAIL__* Due to limitations in our current handling of scoping in spidr, node labels are not contiguous and cannot therefore be list indices. Instead, we use a `List (Nat, Expr)` where the `Nat` is a label for the `Expr` node.

Now, Idris is a purely functional language, which means effects, including state, are explicit. When we build the graph, this state is captured in the `Graph` type constructor, which is essentially a `State` over our topologically-sorted list. Put another way, `Graph` is the _effect_ of adding nodes to a computation graph.

There is both a performance and an ergonomic cost to explicit state, which we discuss in the . This concludes how we construct spidr's internal representation of the tensor graph. Next, we'll take a look at the implication of explicit state for the tensor API, before moving on 

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
