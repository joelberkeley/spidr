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
# The Graph Compiler

_Note: We're not compiler experts, so this tutorial is more about spidr itself than a guide to compiler design._

## Efficiently reusing tensors, and working with `Ref`

spidr explicitly caches tensors so they can be efficiently be reused. Our understanding is that the technique we have used to achieve this is called observable sharing. In this section we discuss what our implementation means for spidr's tensor API.

Caching ensures that the computation you write will be the computation sent to the graph compiler. Unfortunately this comes with downsides. First, there is extra boilerplate. Most tensor operations accept `Tensor shape dtype` and output `Ref (Tensor shape dtype)`, so computations must handle the `Ref` effect. For example, what might be `abs (max x y)` in another library can be `abs !(max !x !y)` in spidr. One notable exception to this is for infix operators where, to avoid unreadable algebra, these accept `Ref (Tensor shape dtype)` values. This means you won't need to write `!(!x * !y) + !z`, but it does also mean you will need to wrap any bare `Tensor shape dtype` values in `pure` to pass it to an infix operator. See this example:
<!-- idris
import Literal
import Tensor
-->
```idris
f : Tensor shape F64 -> Tensor shape F64 -> Ref $ Tensor shape F64
f x y = (abs x + pure y) * pure x
```
Here, `pure` produces a `Ref (Tensor shape F64)` from a `Tensor shape F64`, as does `abs` (the elementwise absolute value function). Addition `(+)` and multiplication `(*)` produce _and accept_ `Ref` so there is no need to wrap the output of `abs x + pure y` in `pure` before passing it to `(*)`. A rule of thumb is that you only need `pure` if both of these are true

* you're passing a value to an infix operator
* the value is either a function argument or is on the left hand side of `x <- expression`

Second, care is needed when reusing expressions to make sure you don't recompute sections of the graph. For example, in
```idris
whoops : Ref $ Tensor [3] S32
whoops = let y = tensor [1, 2, 3]
             z = y + y
          in z * z
```
`z` will be calculated twice, and `y` allocated four times (unless the graph compiler chooses to optimize that out). Instead, we can reuse `z` and `y` with
```idris
ok : Ref $ Tensor [3] S32
ok = do y <- tensor [1, 2, 3]
        z <- (pure y) + (pure y)
        (pure z) * (pure z)
```
Here, `y` and `z` will only be calculated once. This problem can happen more subtley when reusing values from another scope. For example, in
```idris
expensive : Ref $ Tensor [] F64
expensive = reduce @{Sum} [0] !(fill {shape = [100000]} 1.0)

x : Ref $ Tensor [] F64
x = abs !expensive

y : Ref $ Tensor [] F64
y = square !expensive

whoops' : Ref $ Tensor [] F64
whoops' = max !x !y
```
`expensive` is calculated twice. Instead, you could pass the reused part as a function argument
```idris
xf : Tensor [] F64 -> Ref $ Tensor [] F64
xf e = abs e

yf : Tensor [] F64 -> Ref $ Tensor [] F64
yf e = square e

okf : Tensor [] F64 -> Ref $ Tensor [] F64
okf e = max !(xf e) !(yf e)

res : Ref $ Tensor [] F64
res = okf !expensive
```
Note we must pass the `Tensor [] F64` to `xf`, `yf` and `okf`, rather than a `Ref (Tensor [] F64)`, if the tensor is to be reused.
