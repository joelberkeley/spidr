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

## Efficiently reusing tensors

Tensor calculations are not automatically reused in spidr. For example, in
```idris
y : Tensor [] S32
y = let x = 1 + 2 in x + x
```
spidr will interpret each `x` as a different expression, and create two copies of `1 + 2`. This is acceptable for small calculations like this one, but it would be a problem if `x` were expensive to evaluate, or used a lot of space in memory. To prevent recalculating expressions, spidr provides the function
```idris
share : Tensor shape dtype -> Graph $ Tensor shape dtype
```
which binds an expression to a name in the tensor graph. You can efficiently reuse an expression created by `share` as many times as you want. It will only be evaluated once.  This mechanism is called _observable sharing_. In our example, this looks like
```idris
y : Graph $ Tensor [2] F64
y = do
  x <- share $ tensor [1.0, 2.0]
  pure $ x + x 
```
We detail the inner workings of `share` in [How spidr Works](HowSpidrWorks.md).

There are downsides to `share`. First, it's a distraction. We can usually rely on the compiler to reuse expressions efficiently. In `let x : Nat = 1 + 2 in x + x`, Idris reuses the result of `x` without you needing to think about it. But perhaps more importantly, it's possible to forget to use it, and incur a significant performance hit. We are investigating if it is possible to use linear types to catch unintentional tensor reuse at compile time, in an ergonomic way.

The take home message is this: take care not to use any tensor more than once unless you know it has been `share`d.
