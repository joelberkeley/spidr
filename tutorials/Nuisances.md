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
spidr will interpret each `x` as a different expression, and create two copies of `1 + 2`. This is acceptable for small calculations, but it would be a big problem if `x` were expensive to evaluate, or used a lot of space in memory. To prevent recalculating expressions, spidr provides _observable sharing_ via the interface
> ```idris
> interface Shareable a where
>   share : a -> Graph a
> ```
`share` labels all tensor expressions contained within the `a`. You can efficiently reuse a value created by `share` as many times as you like; it will only be evaluated once. In our example, this would be
```idris
y' : Graph $ Tensor [2] F64
y' = do
  x <- share $ tensor [1.0, 2.0]
  pure $ x + x 
```

> *__DETAIL__* Some machine learning compilers, including XLA, will eliminate common subexpressions, so using `share` might not always make a difference. However, eliminating these subexpressions itself requires compute, and even then the compiler might not catch all of them, so we don't recommend relying on this.

There are downsides to `share`. First, it's a distraction. Normally, we can rely on the compiler to reuse expressions by name bindings: in `let x : Nat = 1 + 2 in x + x`, Idris reuses the result of `x` without you needing to think about it. Naturally, we have the same situation in symbolic maths. Perhaps more importantly, though, it's possible to accidentally reuse an expression without sharing it, and thus incur a performance penalty. We are investigating how [linearity](https://www.type-driven.org.uk/edwinb/papers/idris2.pdf) might catch unintentional tensor reuse at compile time.

### Tips for using `share`

* `share` binds values to the scope it is called in. This is important to consider when working with nested functions and currying, particularly when you expect a partially-applied function to be called many times. For example, the program
```idris
add : Tensor [] S32 -> Tensor [] S32 -> Tensor [] S32
add x y = x + y

xs : List (Tensor [] S32)
xs = let sum = 1 + 2
         f = add sum
      in replicate 1000 (f 1)
```
  will calculate `sum` 1000 times. The same problem is observed if we `share` `sum` within the call to `add`
```idris
add' : Tensor [] S32 -> Tensor [] S32 -> Graph $ Tensor [] S32
add' x y = share x <&> \x => x + y

xs' : List (Graph $ Tensor [] S32)
xs' = let sum = 1 + 2
          f = add' sum
       in replicate 1000 (f 1)
```
  as we can infer from the type of xs: we are repeating the effect of sharing `x`, but we should be doing it just once. The solution is to `share` `sum` outside the call to `f`.
```idris
add'' : Tensor [] S32 -> Tensor [] S32 -> Tensor [] S32
add'' x y = x + y

xs'' : Graph $ List (Tensor [] S32)
xs'' = do sum <- share (1 + 2)
          let f = add'' sum
          pure $ replicate 1000 (f 1)
```
* Be aware that if a function on an interface does not return values wrapped in `Graph`, then implementations will not be able to efficiently `share` within the function. As such, it's generally speaking advisable to add `Graph` to interface function return types. The same applies to function aliases.
