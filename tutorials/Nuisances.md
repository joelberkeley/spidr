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
# Nuisances in the Tensor API

## Efficiently reusing tensors with `share`

Tensor calculations are not automatically reused in spidr. For example, in
<!-- idris
import Literal
import Tensor
-->
```idris
y : Tensor [] S32
y = let x = 1 + 2 in x + x
```
spidr will interpret each `x` as a different expression, and create two copies of `1 + 2`. This is acceptable for small calculations like this one, but it would be a big problem if `x` were expensive to evaluate, or used a lot of space in memory. To prevent recalculating expressions, spidr provides _observable sharing_ via the interface
```
interface Shareable a where
  share : a -> Graph a
```
which labels all tensor expressions in an `a`, in the tensor graph. You can efficiently reuse a value created by `share` as many times as you like; it will only be evaluated once. In our example, this would be
```idris
y' : Graph $ Tensor [2] F64
y' = do
  x <- share $ tensor [1.0, 2.0]
  pure $ x + x 
```

> *__DETAIL__* Some machine learning compilers, including XLA, will eliminate common subexpressions, so using `share` might not make all that much difference. However, eliminating common subexpressions will itself use compute, and the compiler might not catch all of them, so we don't recommend relying on this.

There are downsides to `share`. First, it's a distraction. We can usually rely on the compiler to reuse expressions efficiently: in `let x : Nat = 1 + 2 in x + x`, Idris reuses the result of `x` without you needing to think about it. But more importantly, it's possible to forget to use it, and incur a significant performance penalty. We are investigating how to use linear types to catch unintentional tensor reuse at compile time.
