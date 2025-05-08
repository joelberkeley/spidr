<!--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->
# Nuisances in the Tensor API

## Efficiently reusing tensors with `tag`

Tensor calculations are not automatically reused in spidr. For example, in
<!-- idris
import Tensor
-->
```idris
y : Tensor [] S32
y = let x = 1 + 2 in x + x
```
spidr will interpret each `x` as a different expression, and create two copies of `1 + 2`. This is acceptable for small calculations, but it would be a big problem if `x` were expensive to evaluate, or used a lot of space in memory. To prevent recalculating expressions, spidr provides _observable sharing_ via the interface
> ```idris
> interface Taggable a where
>   tag : Monad m => a -> TagT m a
> ```
`tag` tags all tensor expressions contained within the `a`. You can efficiently reuse a value created by `tag` as many times as you like; it will only be evaluated once. In our example, this would be
```idris
y' : Tag $ Tensor [] S32
y' = do
  x <- tag $ 1 + 2
  pure $ x + x 
```
where we've used spidr's convenience alias `Tag = TagT Identity`.

> *__DETAIL__* Some machine learning compilers, including XLA, will eliminate common subexpressions, so using `tag` might not always make a difference. However, eliminating these subexpressions itself requires compute, and even then the compiler might not catch all of them, so we don't recommend relying on this.

There are downsides to `tag`. First, it's a distraction. Normally, we can rely on the compiler to reuse expressions by name bindings: in `let x : Nat = 1 + 2 in x + x`, Idris reuses the result of `x` without you needing to think about it. Naturally, we have the same situation in symbolic maths. Perhaps more importantly, though, it's possible to accidentally reuse an expression without tagging it, and thus incur a performance penalty. We are investigating how [linearity](https://www.type-driven.org.uk/edwinb/papers/idris2.pdf) might catch unintentional tensor reuse at compile time.

### Tips for using `tag`

#### Partially-applied functions

`tag` binds values to the scope it is called in. This is important to consider when working with nested functions and currying, particularly when you expect a partially-applied function to be called many times. For example, the program
```idris
add : Tensor [] S32 -> Tensor [] S32 -> Tensor [] S32
add x y = x + y

bad : List (Tensor [] S32)
bad = let sum = 1 + 2
          f = add sum
       in replicate 1000 (f 1)
```
will calculate `sum` one thousand times. Perhaps counterintuitively, this is _not_ resolved if we tag `sum` within the call to `add`
```idris
addTagged : Tensor [] S32 -> Tensor [] S32 -> Tag $ Tensor [] S32
addTagged x y = tag x <&> \x => x + y

alsoBad : List (Tag $ Tensor [] S32)
alsoBad = let sum = 1 + 2
              f = addTagged sum
           in replicate 1000 (f 1)
```
As we can infer from the type of xs, we are repeatedly tagging `sum`, whilst we mean to tag it once. The solution is to tag `sum` _outside_ the call to `f`.
```idris
good : Tag $ List (Tensor [] S32)
good = do sum <- tag (1 + 2)
          let f = add sum
          pure $ replicate 1000 (f 1)
```
