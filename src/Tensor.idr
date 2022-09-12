{--
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
--}
||| This module contains the `Tensor` object, an array of numbers or booleans, along with a
||| number of functions operating on `Tensor`s.
module Tensor

import Control.Monad.State
import public Data.List
import public Data.List.Elem
import Data.List.Quantifiers
import Decidable.Equality

import Data.Hashable

import Compiler.Eval
import Compiler.Expr
import Compiler.LiteralRW
import Literal
import public Primitive
import public Shape
import public Types
import public Util

----------------------------- core definitions ----------------------------

||| A `Tensor` is a symbolic value, which may refer to either to a scalar value or array of values,
||| though the runtime representation will likely contain more than its value, and will depend on
||| the specific backend.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (0 shape : Shape) -> (0 dtype : Type) -> Type where
  MkTensor : {shape : _} -> Expr shape -> Tensor shape dtype

||| Construct a `Tensor` from `Literal` data.
export
fromLiteral : PrimitiveRW dtype a => {shape : _} -> Literal shape a -> Tensor shape dtype
fromLiteral lit = MkTensor $ FromLiteral {dtype} {shape} lit

namespace F64
  export
  fromDouble : Double -> Tensor [] F64
  fromDouble = fromLiteral . Scalar

namespace S32
  export
  fromInteger : Integer -> Tensor [] S32
  fromInteger = fromLiteral . Scalar . fromInteger

||| Evaluate a `Tensor`, returning its value as a `Literal`. This function builds and executes the
||| computation graph.
|||
||| This function will execute the graph on GPU if one is found, else it will use the host CPU.
|||
||| **Note:**
||| * Each call to `toLiteral` will rebuild and execute the graph. Similarly, multiple calls to 
|||   `toLiteral` on different `Tensor`s in a computation will be treated entirely independently.
|||   `toLiteral` does not store intermediate values. This is a known limitation, and may change in
|||   the future.
||| * `toLiteral` performs logging as a side effect. You can disable this by adjusting the
|||   TensorFlow logging level e.g. with `export TF_CPP_MIN_LOG_LEVEL=3`.
export
toLiteral : PrimitiveRW dtype ty => Tensor shape dtype -> Literal shape ty
toLiteral (MkTensor {shape} expr) = run {dtype} expr

||| A string representation of an unevaluated `Tensor`, detailing all enqueued Xla operations.
||| Useful for debugging.
export
Show (Tensor shape dtype) where
  show (MkTensor expr) = toString expr

||| Finite bounds for numeric tensors.
export
[Finite] Primitive.Num dtype => Bounded (Tensor [] dtype) where
  min = MkTensor $ MinFiniteValue {dtype}
  max = MkTensor $ MaxFiniteValue {dtype}

export
Primitive.Integral a => Cast (Tensor shape a) (Tensor shape F64) where
  cast (MkTensor expr) = MkTensor $ ConvertElementType {dtype=F64} expr 

----------------------------- structural operations ----------------------------

||| Reshape a `Tensor`. For example, `reshape {to=[2, 1]} (fromLiteral [3, 4])` is
||| `fromLiteral [[3], [4]]`. The output can have a different rank to the input.
export
reshape :
  Primitive dtype =>
  {to : _} ->
  {auto 0 sizesEqual : product from = product to} ->
  Tensor from dtype ->
  Tensor to dtype
reshape (MkTensor expr) = MkTensor $ Reshape expr

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified `axis` in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is
||| `fromLiteral [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand :
  Primitive dtype =>
  (axis : Nat) ->
  {auto 0 inBounds : axis `LTE` length shape} ->
  Tensor shape dtype ->
  Tensor (insertAt axis 1 shape) dtype
-- expand axis (MkTensor expr) = MkTensor $ Reshape expr

namespace Squeezable
  ||| A `Squeezable from to` constitutes proof that the shape `from` can be squeezed to the
  ||| shape `to`. Squeezing is the process of removing any number of dimensions of length one.
  public export
  data Squeezable : (0 from : Shape) -> (0 to : Shape) -> Type where
    ||| Proof that a shape can be squeezed to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    Same : Squeezable x x

    ||| Proof that any dimensions (including those of length 1) can be preserved in the process of
    ||| squeezing. For example:
    |||
    ||| ...
    Match : Squeezable from to -> Squeezable (x :: from) (x :: to)

    ||| Proof that any dimensions of length one can be squeezed out. For example:
    |||
    ||| [1, 3, 1, 1, 4] to [3, 4]
    Nest : Squeezable from to -> Squeezable (1 :: from) to

||| Remove dimensions of length one from a `Tensor` such that it has the desired shape. For example:
|||
||| ```idris
||| x : Tensor [2, 1, 3, 1] S32
||| x = fromLiteral [[[[4], [5], [6]]], [[[7], [8], [9]]]]
|||
||| y : Tensor [2, 1, 3] S32
||| y = squeeze x
||| ```
|||
||| is
|||
||| ```idris
||| y : Tensor [2, 1, 3] S32
||| y = fromLiteral [[[4, 5, 6]], [[7, 8, 9]]]
||| ```
export
squeeze :
  Primitive dtype =>
  {to : _} ->
  {auto 0 shapesSqueezable : Squeezable from to} ->
  Tensor from dtype ->
  Tensor to dtype
-- squeeze (MkTensor expr) = MkTensor $ Reshape expr 

||| A `SliceOrIndex d` is a valid slice or index into a dimension of size `d`. See `slice` for
||| details.
export
data SliceOrIndex : Nat -> Type where
  Slice :
    (from, to : Nat) ->
    {size : _} ->
    {auto 0 fromTo : from + size = to} ->
    {auto 0 inDim : LTE to d} ->
    SliceOrIndex d
  Index : (idx : Nat) -> {auto 0 inDim : LT idx d} -> SliceOrIndex d
  DynamicSlice : Tensor [] U64 -> (size : Nat) -> {auto 0 inDim : LTE size d} -> SliceOrIndex d
  DynamicIndex : Tensor [] U64 -> SliceOrIndex d

||| Index at `idx`. See `slice` for details.
public export
at : (idx : Nat) -> {auto 0 inDim : LT idx d} -> SliceOrIndex d
at = Index

namespace Dynamic
  ||| Index at the specified index. See `slice` for details.
  public export
  at : Tensor [] U64 -> SliceOrIndex d
  at = DynamicIndex

||| Slice from `from` (inclusive) to `to` (exclusive). See `slice` for details.
public export
(.to) :
  (from, to : Nat) ->
  {size : _} ->
  {auto 0 fromTo : from + size = to} ->
  {auto 0 inDim : LTE to d} ->
  SliceOrIndex d
(.to) = Slice

||| Slice `size` elements starting at the specified scalar `U64` index. See `slice` for details.
public export
(.size) : Tensor [] U64 -> (size : Nat) -> {auto 0 inDim : LTE size d} -> SliceOrIndex d
(.size) = DynamicSlice

||| Slice across all indices along an axis. See `slice` for details.
public export
all : {d : _} -> SliceOrIndex d
all = Slice 0 @{%search} @{reflexive {ty=Nat}} d

||| A `MultiSlice shape` is a valid multi-dimensionsal slice into a tensor with shape `shape`.
||| See `slice` for details.
public export
data MultiSlice : Shape -> Type where
  Nil : MultiSlice ds
  (::) : SliceOrIndex d -> MultiSlice ds -> MultiSlice (d :: ds)

namespace MultiSlice
  ||| The shape of a tensor produced by slicing with the specified multi-dimensional slice. See
  ||| `Tensor.slice` for details.
  public export
  slice : {shape : _} -> MultiSlice shape -> Shape
  slice {shape} [] = shape
  slice {shape=(_ :: _)} (Slice {size} _ _ :: xs) = size :: slice xs
  slice {shape=(_ :: _)} (Index _ :: xs) = slice xs
  slice {shape=(_ :: _)} (DynamicSlice _ size :: xs) = size :: slice xs
  slice {shape=(_ :: _)} (DynamicIndex _ :: xs) = slice xs

||| Slice or index `Tensor` axes. Each axis can be sliced or indexed, and this can be done with
||| either static (`Nat`) or dynamic (scalar `U64`) indices.
|||
||| **Static indices**
|||
||| Static indices are `Nat`s. For example, for
||| ```
||| x : Tensor [5, 6] S32
||| x = fromLiteral [
|||       [ 0,  1,  2,  3,  4,  5],
|||       [ 6,  7,  8,  9, 10, 11],
|||       [12, 13, 14, 15, 16, 17],
|||       [18, 19, 20, 21, 22, 23],
|||       [24, 25, 26, 27, 28, 29]
|||     ]
||| ```
||| we can index as `slice [at 1] x` to get
||| ```
||| x : Tensor [6] S32
||| x = fromLiteral [6, 7, 8, 9, 10, 11]
||| ```
||| or we can slice as `slice [2.to 4] x` to get
||| ```
||| x : Tensor [2, 6] S32
||| x = fromLiteral [
|||       [12, 13, 14, 15, 16, 17],
|||       [18, 19, 20, 21, 22, 23]
|||     ]
||| ```
||| Note that in `2.to 4`, the 2 is inclusive, and the 4 exclusive, so we return indices 2 and 3.
|||
||| **Dynamic indices**
|||
||| Dynamic indices are scalar `U64` values, and the API works slightly differently because we
||| can't know the value of dynamic indices until the graph is executed. For indexing, with scalar
||| `U64` index `i` in `slice [at i] x`, `i` is clamped to be a valid index into that dimension.
||| For example, for `i = fromLiteral 1`, `slice [at i] x` is
||| ```
||| x : Tensor [6] S32
||| x = fromLiteral [6, 7, 8, 9, 10, 11]
||| ```
||| as in the static case. However, for `i = fromLiteral 10`, `slice [at i] x` returns the last row
||| ```
||| x : Tensor [6] S32
||| x = fromLiteral [24, 25, 26, 27, 28, 29]
||| ```
||| We can also slice by specifying a scalar `U64` start index, and a static size, as
||| `slice [i.size 2] x` with `i = fromLiteral 2` to get
||| ```
||| x : Tensor [2, 6] S32
||| x = fromLiteral [
|||       [12, 13, 14, 15, 16, 17],
|||       [18, 19, 20, 21, 22, 23]
|||     ]
||| ```
||| For a given slice `size`, the dynamic start index is clamped such that we always get `size`
||| elements along that axis. For example, `slice [i.size 2] x` with `i = fromLiteral 4` is
||| ```
||| x : Tensor [2, 6] S32
||| x = fromLiteral [
|||       [18, 19, 20, 21, 22, 23],
|||       [24, 25, 26, 27, 28, 29]
|||     ]
||| ```
||| which starts at index 3 rather than index 4.
|||
||| **Mixed static, dynamic, slicing and indexing**
|||
||| Each axis can only be sliced or indexed, and must use only static or dynamic indices. However,
||| across axes, we can mix these four arbitrarily. For example, with `slice [2.to 4, at 1] x` to
||| get
||| ```
||| x : Tensor [2] S32
||| x = fromLiteral [13, 19]
||| ```
||| or with `i = fromLiteral 2` in `slice [at 1, i.size 2] x` to get
||| ```
||| x : Tensor [2] S32
||| x = fromLiteral [7, 8]
||| ```
|||
||| Slices and indices apply to the leading axes of the tensor. For trailing axes omitted from the
||| multi-dimensional slice, the whole of the axis is returned. If we want to slice or index over
||| later axes and retain all indices in a leading axis, we can use the convenience function `all`,
||| as `slice [all, at 3] x` to get
||| ```
||| x : Tensor [5] S32
||| x = fromLiteral [[3], [9], [15], [21], [27]]
||| ```
||| This is exactly the same as the more manual `slice [0.to 5, at 3] x` and
||| `slice [(fromLiteral 0).size 5, at 3] x`.
|||
||| @at The multi-dimensional slices and indices at which to slice the tensor.
export
slice : Primitive dtype => (at : MultiSlice shape) -> Tensor shape dtype -> Tensor (slice at) dtype
-- slice at (MkTensor expr) =
--   let sliced = Slice (mapd start (const 0) at) (mapd stop id at) (replicate (length shape) 1) expr
--       sliced = DynamicSlice (mapd dynStart (const zero) at) (mapd size id at) sliced
--    in MkTensor $ Reshape sliced

--       where
--       mapd :
--         ((Nat -> a) -> {d : Nat} -> SliceOrIndex d -> a) ->
--         (Nat -> a) ->
--         {shape : Shape} ->
--         MultiSlice shape ->
--         List a
--       mapd _ dflt {shape} [] = Prelude.map dflt shape
--       mapd f dflt (x :: xs) = f dflt x :: mapd f dflt xs

--       start : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
--       start _ (Slice from _) = from
--       start _ (Index idx) = idx
--       start f {d} _ = f d

--       stop : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
--       stop _ (Slice _ to) = to
--       stop _ (Index idx) = S idx
--       stop f {d} _ = f d

--       zero : Expr []
--       zero = FromLiteral {dtype=U64} 0

--       dynStart : (Nat -> Expr) -> {d : Nat} -> SliceOrIndex d -> Expr
--       dynStart _ (DynamicSlice (MkTensor from) _) = from
--       dynStart _ (DynamicIndex (MkTensor idx)) = idx
--       dynStart f {d} _ = f d

--       size : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
--       size _ (Slice {size=size'} _ _) = size'
--       size _ (Index _) = 1
--       size _ (DynamicSlice _ size') = size'
--       size _ (DynamicIndex _) = 1

||| Concatenate two `Tensor`s along the specfied `axis`. For example,
||| `concat 0 (fromLiteral [[1, 2], [3, 4]]) (fromLiteral [[5, 6]])` and
||| `concat 1 (fromLiteral [[3], [6]]) fromLiteral ([[4, 5], [7, 8]])` are both
||| `fromLiteral [[1, 2], [3, 4], [5, 6]]`.
export
concat :
  Primitive dtype =>
  (axis' : Nat) ->
  Tensor front' dtype ->
  Tensor end' dtype ->
  {auto 0 inBoundsf : InBounds axis' front'} ->
  {auto 0 inBoundse : InBounds axis' end'} ->
  {auto 0 shapesConcatenable : deleteAt axis' front' = deleteAt axis' end'} ->
  Tensor (replaceAt axis' (index axis' front' + index axis' end') front') dtype
concat axis (MkTensor expr) (MkTensor expr') = MkTensor $ Concat axis expr expr'

||| The diagonal of a matrix as a vector. For example, for
||| ```
||| x : Tensor [3, 3] S32
||| x = fromLiteral [[0, 1, 2],
|||                  [3, 4, 5],
|||                  [6, 7, 8]]
||| ```
||| `diag x` is `fromLiteral [0, 4, 8]`.
export
diag : Primitive dtype => Tensor [n, n] dtype -> Tensor [n] dtype
diag (MkTensor expr) = MkTensor $ Diag {leading=[]} {n} expr

||| Represents the upper- or lower-trinagular component of a matrix.
public export
data Triangle = Upper | Lower

||| Get the upper- or lower-triangular component of a matrix. For example, for
||| ```
||| x : Tensor [3, 3] S32
||| x = fromLiteral [[1, 2, 3],
|||                  [4, 5, 6],
|||                  [7, 8, 9]]
||| ```
||| `triangle Lower x` is
||| ```
||| x : Tensor [3, 3] S32
||| x = fromLiteral [[1, 0, 0],
|||                  [4, 5, 0],
|||                  [7, 8, 9]]
||| ```
export
triangle : Primitive dtype => Triangle -> Tensor [n, n] dtype -> Tensor [n, n] dtype
triangle tri (MkTensor expr) =
  MkTensor $ Triangle (case tri of Upper => False; Lower => True) {leading=[]} {n} expr 

||| Tranpose a matrix. For example, `(fromLiteral [[1, 2], [3, 4]]).T` is
||| `fromLiteral [[1, 3], [2, 4]]`.
export
(.T) : Tensor [m, n] dtype -> Tensor [n, m] dtype
-- (MkTensor expr).T = MkTensor $ Transpose [1, 0] expr

||| Transpose axes of a tensor. This is a more general version of `(.T)`, in which you can transpose
||| any number of axes in a tensor of arbitrary rank. The i'th axis in the resulting tensor
||| corresponds to the `index i ordering`'th axis in the input tensor. For example, for
||| ```
||| x : Tensor [2, 3, 4] S32
||| x = fromLiteral [[[ 0,  1,  2,  3],
|||                   [ 4,  5,  6,  7],
|||                   [ 8,  9, 10, 11]],
|||                  [[12, 13, 14, 15],
|||                   [16, 17, 18, 19],
|||                   [20, 21, 22, 23]]]
||| ```
||| `transpose [0, 2, 1]` is
||| ```
||| x : Tensor [2, 4, 3] S32
||| x = fromLiteral [[[ 0,  4,  8],
|||                   [ 1,  5,  9],
|||                   [ 2,  6, 10],
|||                   [ 3,  7, 11]],
|||                  [[12, 16, 20],
|||                   [13, 17, 21],
|||                   [14, 18, 22],
|||                   [15, 19, 23]]]
||| ```
||| `transpose [2, 0, 1]` is
||| ```
||| x : Tensor [4, 2, 3] S32
||| x = fromLiteral [[[ 0,  4,  8],
|||                   [12, 16, 20]],
|||                  [[ 1,  5,  9],
|||                   [13, 17, 21]],
|||                  [[ 2,  6, 10],
|||                   [14, 18, 22]],
|||                  [[ 3,  7, 11],
|||                   [15, 19, 23]]]
||| ```
|||
||| In order to see what effect transposing a tensor has, it can help to bear in mind the following:
||| * if an element can be found with `slice [at 3, at 4, at 5] x` in the original tensor,
|||   that same element can instead be found with `slice [at 5, at 3, at 4]` given a
|||   `transpose [2, 0, 1]`. That is, transposing axes re-orders indices when indexing.
||| * with `transpose [2, 0, 1]`, traversing the first axis in the result is equivalent to
|||   traversing the last axis in the input. Similarly, traversing the last axis in the result is
|||   equivalent to traversing the second axis in the input.
export
transpose :
  (ordering : List Nat) ->
  Tensor shape dtype ->
  {auto 0 lengths : length ordering = length shape} ->
  {auto 0 unique : Sorted Neq ordering} ->
  {auto 0 inBounds : All (flip InBounds shape) ordering} ->
  Tensor (map (dflip List.index shape) ordering) dtype
-- transpose ordering (MkTensor expr) = MkTensor $ Transpose ordering expr

||| The identity tensor, with inferred shape and element type. For example,
||| ```
||| x : Tensor [2, 2] S32
||| x = identity
||| ```
||| is
||| ```
||| x : Tensor [2, 2] S32
||| x = [[1, 0],
|||      [0, 1]]
||| ```
export
identity : Primitive.Num dtype => {n : _} -> Tensor [n, n] dtype
identity = MkTensor $ Identity {n, dtype}

||| Broadcast a `Tensor` to a new compatible shape. For example,
|||
||| ```idris
||| x : Tensor [2, 3] S32
||| x = broadcast (fromLiteral [4, 5, 6])
||| ```
|||
||| is
|||
||| ```idris
||| x : Tensor [2, 3] S32
||| x = fromLiteral [[4, 5, 6], [4, 5, 6]]
||| ```
export
broadcast :
  Primitive dtype =>
  {to : _} ->
  {auto shapesOK : Broadcastable from to} ->
  Tensor from dtype ->
  Tensor to dtype
broadcast xs with (xs)
  _ | (MkTensor {shape=f} expr) = MkTensor $ Broadcast {from=f, dtype} expr

%hint
export
scalarToAnyOk : (to : Shape) -> Broadcastable [] to
scalarToAnyOk [] = Same
scalarToAnyOk (_ :: xs) = Nest (scalarToAnyOk xs)

||| A `Tensor` where every element has the specified value. For example,
|||
||| ```idris
||| fives : Tensor [2, 3] S32
||| fives = fill 5
||| ```
||| is
||| ```idris
||| fives : Tensor [2, 3] S32
||| fives = fromLiteral [[5, 5, 5], [5, 5, 5]]
||| ```
export
fill : PrimitiveRW dtype ty => {shape : _} -> ty -> Tensor shape dtype
fill = broadcast {shapesOK=scalarToAnyOk shape} . fromLiteral . Scalar

----------------------------- generic operations ----------------------------

{-
||| Lift a unary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| recip : Tensor [] F64 -> Tensor [] F64
||| recip = (1.0 /)
||| ```
||| can be lifted to an element-wise reciprocal function as `map recip (fromLiteral [-2, 0.4])`,
||| which is `fromLiteral [-0.5, 2.5]`.
export
map : (Primitive a, Primitive b) => (Tensor [] a -> Tensor [] b) -> Tensor shape a -> Tensor shape b
map f (MkTensor {shape} expr) =
  let p0 = Parameter 0 {shape=[]} "" {dtype=a}
      MkTensor exprf = f (MkTensor p0)
   in MkTensor $ Map (MkFn [p0] exprf) [expr] (range $ length shape)

||| Lift a binary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| addRecip : Tensor [] F64 -> Tensor [] F64 -> Tensor [] F64
||| addRecip x y = x + 1.0 / y
||| ```
||| can be lifted to an element-wise function as
||| `map2 addRecip (fromLiteral [3.0, -3.0]) (fromLiteral [-2.0, 0.4])`, which is
||| `fromLiteral [2.5, -0.5]`.
export
map2 :
  (Primitive a, Primitive b, Primitive c) =>
  (Tensor [] a -> Tensor [] b -> Tensor [] c) ->
  Tensor shape a ->
  Tensor shape b ->
  Tensor shape c
map2 f (MkTensor {shape} expr0) (MkTensor expr1) =
  let p0 = Parameter 0 {shape=[]} "" {dtype=a}
      p1 = Parameter 1 {shape=[]} "" {dtype=b}
      MkTensor exprf = f (MkTensor p0) (MkTensor p1)
   in MkTensor $ Map (MkFn [p0, p1] exprf) [expr0, expr1] (range $ length shape)
-}

||| Reduce elements along one `axis` of a `Tensor` according to a specified `reducer` `Monoid`.
||| For example, if `x = fromLiteral [[0, 1, 2], [3, 4, 5]]`, then reduce @{Sum} 0 x` is
||| `fromLiteral [3, 5, 7]` and `reduce @{Sum} 1 x` to `fromLiteral [3, 12]`.
|||
||| @reducer How to reduce elements along the given `axis`.
||| @axis The axis along which to reduce elements.
export
reduce :
  (reducer : Monoid (Tensor [] dtype)) =>
  Primitive dtype =>
  (axes : List Nat) ->
  {auto 0 axesUnique : Sorted LT axes} ->
  {auto 0 axesInBounds : All (flip InBounds shape) axes} ->
  Tensor shape dtype ->
  Tensor (deleteAt axes shape) dtype
reduce axes (MkTensor expr) =
  let semigroup : Monoid a -> Semigroup a
      semigroup _ = %search

      p0 := Parameter 0 {shape=[]} "" {dtype}
      p1 := Parameter 1 {shape=[]} "" {dtype}
      MkTensor exprf := (<+>) @{semigroup reducer} (MkTensor p0) (MkTensor p1)
      MkTensor neutral := neutral @{reducer}
   in MkTensor $ Reduce (p0, p1, exprf) neutral axes expr

||| Sort the elements of a `Tensor` along a specified `dimension` according to a scalar-wise
||| ordering. For sorting function `f`, elements are sorted such that for consecutive sorted
||| elements `a` and `b`, either `f a b` is true, or `f a b` *and* `f b a` are false.
|||
||| **Note:** Sorting is not stable, meaning elements that compare equal according the ordering may
||| be sorted in a different order to the order they appear in the input.
|||
||| For example, for `x = fromLiteral [1, 3, 4, 2]`, `sort (<) 0 x` is
||| `fromLiteral [[1, 2, 4], [3, 6, 5]]` and `sort (<) 1 x` is `fromLiteral [[1, 4, 6], [2, 3, 5]]`.
export
sort :
  Primitive dtype =>
  (Tensor [] dtype -> Tensor [] dtype -> Tensor [] PRED) ->
  (dimension : Nat) ->
  Tensor shape dtype ->
  {auto 0 dimInBounds : InBounds dimension shape} ->
  Tensor shape dtype
sort comp dimension (MkTensor expr) =
  let p0 = Parameter 0 {shape=[]} "" {dtype}
      p1 = Parameter 1 {shape=[]} "" {dtype}
      MkTensor exprComp = comp (MkTensor p0) (MkTensor p1)
   in MkTensor $ Sort (p0, p1, exprComp) dimension False [expr]

||| Reverse elements along the specified axes. For example, for
||| ```
||| x : Tensor [2, 3] S32
||| x = fromLiteral [
|||   [-2, -1,  0],
|||   [ 1,  2,  3]
||| ]
||| ```
||| `reverse [0] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = fromLiteral [
|||   [ 1,  2,  3]
|||   [-2, -1,  0],
||| ]
||| ```
||| `reverse [1] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = fromLiteral [
|||   [ 0, -1, -2],
|||   [ 3,  2,  1]
||| ]
||| ```
||| and `reverse [0, 1] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = fromLiteral [
|||   [ 3,  2,  1]
|||   [ 0, -1, -2],
||| ]
||| ```
|||
||| **Note:** This function requires `axes` is ordered simply so that elements are unique.
||| The ordering itself is irrelevant to the implementation, but ensures uniqueness without using
||| proofs of contradiction that can be difficult for Idris to construct.
export
reverse :
  (axes : List Nat) ->
  {auto 0 axesUnique : Sorted LT axes} ->
  {auto 0 axesInBounds : All (flip InBounds shape) axes} ->
  Tensor shape dtype ->
  Tensor shape dtype
reverse axes (MkTensor expr) = MkTensor $ Reverse axes expr 

----------------------------- numeric operations ----------------------------

||| `fromLiteral [True, False]`.
export
(==) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) == (MkTensor exprr) = MkTensor $ Eq exprl exprr

||| Element-wise inequality. For example, `fromLiteral [1, 2] /= fromLiteral [1, 3]` is
||| `fromLiteral [False, True]`.
export
(/=) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) /= (MkTensor exprr) = MkTensor $ Ne exprl exprr

||| Element-wise less than. For example, `fromLiteral [1, 2, 3] < fromLiteral [2, 2, 2]` is
||| `fromLiteral [True, False, False]`.
export
(<) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) < (MkTensor exprr) = MkTensor $ Lt exprl exprr

||| Element-wise greater than. For example, `fromLiteral [1, 2, 3] > fromLiteral [2, 2, 2]` is
||| `fromLiteral [False, False, True]`.
export
(>) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) > (MkTensor exprr) = MkTensor $ Gt exprl exprr

||| Element-wise less than or equal. For example, `fromLiteral [1, 2, 3] <= fromLiteral [2, 2, 2]`
||| is `fromLiteral [True, True, False]`.
export
(<=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) <= (MkTensor exprr) = MkTensor $ Le exprl exprr

||| Element-wise greater than or equal. For example,
||| `fromLiteral [1, 2, 3] >= fromLiteral [2, 2, 2]` is `fromLiteral [False, True, True]`.
export
(>=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(MkTensor exprl) >= (MkTensor exprr) = MkTensor $ Ge exprl exprr

||| Element-wise boolean and. For example,
||| `fromLiteral [True, True, False, False] && fromLiteral [True, False, True, False]` is
||| `fromLiteral [True, False, False, False]`.
export
(&&) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(MkTensor exprl) && (MkTensor exprr) = MkTensor $ And exprl exprr

namespace Semigroup
  export
  [All] Semigroup (Tensor shape PRED) where
    (<+>) = (&&)

namespace Monoid
  export
  [All] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.All where
    neutral = fill True

||| Element-wise boolean or. For example,
||| `fromLiteral [True, True, False, False] || fromLiteral [True, False, True, False]` is
||| `fromLiteral [True, True, True, False]`.
export
(||) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(MkTensor exprl) || (MkTensor exprr) = MkTensor $ Or exprl exprr

namespace Semigroup
  export
  [Any] Semigroup (Tensor shape PRED) where
    (<+>) = (||)

namespace Monoid
  export
  [Any] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.Any where
    neutral = fill False

||| Element-wise boolean negation. For example, `not (fromLiteral [True, False])` is
||| `fromLiteral [False, True]`.
export
not : Tensor shape PRED -> Tensor shape PRED
not (MkTensor expr) = MkTensor $ Not expr

||| Choose elements from two `Tensor`s based on a `Tensor` of predicates. For each element in the
||| predicates, the output will use the corresponding element from `onTrue` if the element is
||| truthy, else the element from `onFalse`. For example, for
||| ```
||| preds : Tensor [3] PRED
||| preds = fromLiteral [False, True, False]
|||
||| onTrue : Tensor [3] S32
||| onTrue = fromLiteral [1, 2, 3]
|||
||| onFalse : Tensor [3] S32
||| onFalse = fromLiteral [4, 5, 6]
||| ```
||| `select preds onTrue onFalse` is `fromLiteral [4, 2, 6]`.
|||
||| @onTrue The elements to choose where the predicate elements are truthy.
||| @onFalse The elements to choose where the predicate elements are falsy.
export
select :
  Primitive dtype =>
  Tensor shape PRED ->
  (onTrue : Tensor shape dtype) ->
  (onFalse : Tensor shape dtype) ->
  Tensor shape dtype
select (MkTensor pred) (MkTensor true) (MkTensor false) = MkTensor $ Select pred true false

||| Use a scalar predicate to choose which of two functions to evaluate. If the predicte is truthy,
||| evaluate `onTrue` on the corresponding specified argument, otherwise evaluate `onFalse` on the
||| corresponding specified argument. The result of the evaluated function is returned. For example,
||| for
||| ```
||| x : Tensor [2] S32
||| x = fromLiteral [2, -1]
|||
||| y : Tensor [2, 2] S32
||| y = fromLiteral [[5, 6],
|||                  [7, 8]]
||| ```
||| `cond (fromLiteral True) (fromLiteral 2 *) x diag y` is `fromLiteral [4, -2]` and
||| `cond (fromLiteral False) (fromLiteral 2 *) x diag y` to `fromLiteral [5, 8]`.
|||
||| While both functions will be called for the purposes of defining the computation, only one will
||| be evaluated with its specified argument. That is, this function short-circuits.
|||
||| @onTrue The function to execute if the predicate is truthy.
||| @onFalse The function to execute if the predicate is falsy.
export
cond :
  (Primitive tt, Primitive ft, Primitive dtype) =>
  {shape, ts, fs : _} ->
  Tensor [] PRED ->
  (onTrue : Tensor ts tt -> Tensor shape dtype) -> Tensor ts tt ->
  (onFalse : Tensor fs ft -> Tensor shape dtype) -> Tensor fs ft ->
  Tensor shape dtype
cond (MkTensor pred) onTrue (MkTensor true) onFalse (MkTensor false) =
    let pt = Parameter 0 {shape=ts} "" {dtype=tt}
        pf = Parameter 0 {shape=fs} "" {dtype=ft}
        MkTensor exprTrue = onTrue (MkTensor pt)
        MkTensor exprFalse = onFalse (MkTensor pf)
     in MkTensor $ Cond pred (pt, exprTrue) true (pf, exprFalse) false

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

namespace Vector
  ||| Vector dot product with a tensor of any rank. The vector dot product is with the first axis of
  ||| the right-hand side tensor. For example `fromLiteral [0, 1, 2] @@ fromLiteral [-1, -3, -1]` is
  ||| `-1`.
  |||
  ||| **WARNING** Not well tested
  export
  (@@) : Primitive.Num dtype => Tensor [S m] dtype -> Tensor [S m] dtype -> Tensor [] dtype
  (MkTensor l) @@ (MkTensor r) = MkTensor $ Dot l r

namespace Matrix
  ||| Matrix multiplication with a matrix or vector. Contraction is along the last axis of the first
  ||| and the first axis of the last. For example:
  |||
  ||| ```idris
  ||| x : Tensor [2, 3] S32
  ||| x = fromLiteral [[-1, -2, -3], [0, 1, 2]]
  |||
  ||| y : Tensor [3, 1] S32
  ||| y = fromLiteral [[4, 0, 5]]
  |||
  ||| z : Tensor [2, 1] S32
  ||| z = x @@ y
  ||| ```
  |||
  ||| is
  |||
  ||| ```idris
  ||| z : Tensor [2, 1] S32
  ||| z = fromLiteral [-19, 10]
  ||| ```
  |||
  ||| **WARNING** Not well tested
  export
  (@@) :
    (Primitive dtype, Primitive.Num dtype) =>
    Tensor [n, S m] dtype ->
    Tensor (S m :: tl) dtype ->
    {auto 0 vectorTail : length tl `LTE` 1} ->
    Tensor (n :: tl) dtype
  (MkTensor l) @@ (MkTensor r) = MkTensor $ Dot l r 

||| Element-wise addition. For example, `fromLiteral [1, 2] + fromLiteral [3, 4]` is
||| `fromLiteral [4, 6]`.
export
(+) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(MkTensor exprl) + (MkTensor exprr) = MkTensor $ Add exprl exprr

namespace Semigroup
  export
  [Sum] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (+)

namespace Monoid
  export
  [Sum] {shape : _} ->
        Prelude.Num a =>
        PrimitiveRW dtype a =>
        Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Sum where
      neutral = fill 0

||| Element-wise negation. For example, `- fromLiteral [1, -2]` is `fromLiteral [-1, 2]`.
export
negate : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype
negate (MkTensor expr) = MkTensor $ Neg expr

||| Element-wise subtraction. For example, `fromLiteral [3, 4] - fromLiteral [4, 2]` is
||| `fromLiteral [-1, 2]`.
export
(-) : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(MkTensor exprl) - (MkTensor exprr) = MkTensor $ Sub exprl exprr

||| Element-wise multiplication. For example, `fromLiteral [2, 3] * fromLiteral [4, 5]` is
||| `fromLiteral [8, 15]`.
export
(*) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(MkTensor exprl) * (MkTensor exprr) = MkTensor $ Mul exprl exprr

namespace Scalarwise
  ||| Multiplication by a scalar. For example, `fromLiteral 2 * fromLiteral [3, 5]` is
  ||| `fromLiteral [6, 10]`.
  |||
  ||| The RHS is required to be non-scalar simply to avoid ambiguities with element-wise `(*)`.
  export
  (*) : Primitive.Num dtype => Tensor [] dtype -> Tensor (d :: ds) dtype -> Tensor (d :: ds) dtype
  l * r with (r)
    _ | (MkTensor {shape=_ :: _} _) = (broadcast {shapesOK=scalarToAnyOk (d :: ds)} l) * r

namespace Semigroup
  export
  [Prod] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (*)

namespace Monoid
  export
  [Prod]
      {shape : _} ->
      Prelude.Num a =>
      PrimitiveRW dtype a =>
      Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Prod where
      neutral = fill 1

||| Element-wise floating point division. For example, `fromLiteral [2, 3] / fromLiteral [4, 5]` is
||| `fromLiteral [0.5, 0.6]`.
export
(/) : Primitive.Fractional dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(MkTensor exprl) / (MkTensor exprr) = MkTensor $ Div exprl exprr

namespace Scalarwise
  ||| Floating point division by a scalar. For example, `fromLiteral [3.4, -5.6] / fromLiteral 2` is
  ||| `fromLiteral [1.7, -2.8]`.
  |||
  ||| The LHS is required to be non-scalar simply to avoid ambiguities with element-wise `(/)`.
  export
  (/) :
    Primitive.Fractional dtype =>
    Tensor (d :: ds) dtype ->
    Tensor [] dtype ->
    Tensor (d :: ds) dtype
  l / r with (l)
    _ | (MkTensor {shape=_ :: _} _) = l / (broadcast {shapesOK=scalarToAnyOk (d :: ds)} r)

||| The element-wise reciprocal. For example, `recip (fromLiteral [-2, 0, 0.2])`
||| is `fromLiteral [-0.5, nan, 5]`.
export
recip : Tensor shape F64 -> Tensor shape F64
recip (MkTensor expr) = MkTensor $ Reciprocal expr

infixr 9 ^

||| Each element in `base` raised to the power of the corresponding element in `exponent`.
||| example, `fromLiteral [2, 25, -9] ^ fromLiteral [3, -0.5, 0.5]` is `fromLiteral [8, 0.2, nan]`.
|||
||| Note: The behaviour of this function is not well-defined at negative or positive infinity, or
|||   NaN.
|||
||| Note: The first root is used.
export
(^) : Tensor shape F64 -> Tensor shape F64 -> Tensor shape F64
(MkTensor exprl) ^ (MkTensor exprr) = MkTensor $ Pow exprl exprr

||| Element-wise absolute value. For example, `abs (fromLiteral [-2, 3])` is
||| `fromLiteral [2, 3]`.
export
abs : Primitive.Abs dtype => Tensor shape dtype -> Tensor shape dtype
abs (MkTensor expr) = MkTensor $ Abs expr

||| The element-wise natural exponential. For example, `exp (fromLiteral [-1, 0, 2])` is
||| `fromLiteral [1 / euler, 1, pow euler 2]`.
export
exp : Tensor shape F64 -> Tensor shape F64
exp (MkTensor expr) = MkTensor $ Exp expr

||| The element-wise floor function. For example,
||| `floor (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floor : Tensor shape F64 -> Tensor shape F64
floor (MkTensor expr) = MkTensor $ Floor expr

||| The element-wise ceiling function. For example,
||| `ceil (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceil : Tensor shape F64 -> Tensor shape F64
ceil (MkTensor expr) = MkTensor $ Ceil expr

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `log (fromLiteral [1 / euler, 1, euler * euler])` is `fromLiteral [-1, 0, 2]`.
export
log : Tensor shape F64 -> Tensor shape F64
log (MkTensor expr) = MkTensor $ Log expr

||| The element-wise logistic function equivalent to `1 / 1 + exp (-x)`.
export
logistic : Tensor shape F64 -> Tensor shape F64
logistic (MkTensor expr) = MkTensor $ Logistic expr

||| The element-wise sine.
export
sin : Tensor shape F64 -> Tensor shape F64
sin (MkTensor expr) = MkTensor $ Sin expr

||| The element-wise cosine.
export
cos : Tensor shape F64 -> Tensor shape F64
cos (MkTensor expr) = MkTensor $ Cos expr

||| The element-wise tangent.
export
tan : Tensor shape F64 -> Tensor shape F64
tan (MkTensor expr) = MkTensor $ Tan expr

||| The element-wise inverse sine.
export
asin : Tensor shape F64 -> Tensor shape F64
asin (MkTensor expr) = MkTensor $ Asin expr

||| The element-wise inverse cosine.
export
acos : Tensor shape F64 -> Tensor shape F64
acos (MkTensor expr) = MkTensor $ Acos expr

||| The element-wise inverse tangent.
export
atan : Tensor shape F64 -> Tensor shape F64
atan (MkTensor expr) = MkTensor $ Atan expr

||| The element-wise hyperbolic sine.
export
sinh : Tensor shape F64 -> Tensor shape F64
sinh (MkTensor expr) = MkTensor $ Sinh expr

||| The element-wise hyperbolic cosine.
export
cosh : Tensor shape F64 -> Tensor shape F64
cosh (MkTensor expr) = MkTensor $ Cosh expr

||| The element-wise hyperbolic tangent.
export
tanh : Tensor shape F64 -> Tensor shape F64
tanh (MkTensor expr) = MkTensor $ Tanh expr

||| The element-wise inverse hyperbolic sine.
export
asinh : Tensor shape F64 -> Tensor shape F64
asinh (MkTensor expr) = MkTensor $ Asinh expr

||| The element-wise inverse hyperbolic cosine.
export
acosh : Tensor shape F64 -> Tensor shape F64
acosh (MkTensor expr) = MkTensor $ Acosh expr

||| The element-wise inverse hyperbolic tangent.
export
atanh : Tensor shape F64 -> Tensor shape F64
atanh (MkTensor expr) = MkTensor $ Atanh expr

||| An approximation to the element-wise error function.
export
erf : Tensor shape F64 -> Tensor shape F64
erf (MkTensor expr) = MkTensor $ Erf expr

||| The element-wise square. For example, `square (fromLiteral [-2, 0, 3])`
||| is `fromLiteral [4, 0, 9]`.
export
square : Tensor shape F64 -> Tensor shape F64
square (MkTensor expr) = MkTensor $ Square expr

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrt (fromLiteral [0, 9])` is `fromLiteral [0, 3]`.
export
sqrt : Tensor shape F64 -> Tensor shape F64
sqrt (MkTensor expr) = MkTensor $ Sqrt expr

||| The element-wise minimum of the first argument compared to the second. For example,
||| `min (fromLiteral [-3, -1, 3]) (fromLiteral [-1, 0, 1])` is `fromLiteral [-3, -1, 1]`.
|||
||| **Note:** There is a known issue where sometimes the wrong value is chosen if one value is NaN.
export
min : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
min (MkTensor exprl) (MkTensor exprr) = MkTensor $ Min exprl exprr

||| **Note:** There is a known issue where sometimes the wrong value is chosen if one value is NaN.
namespace Semigroup
  export
  [Min] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = min

namespace Monoid
  export
  [Min] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Min where
      neutral = fill (1.0 / 0.0)

||| The element-wise maximum of the first argument compared to the second. For example,
||| `max (fromLiteral [-3, -1, 3]) (fromLiteral [-1, 0, 1])` is `fromLiteral [-1, 0, 3]`.
|||
||| **Note:** There is a known issue where sometimes the wrong value is chosen if one value is NaN.
export
max : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
max (MkTensor exprl) (MkTensor exprr) = MkTensor $ Max exprl exprr

||| **Note:** There is a known issue where sometimes the wrong value is chosen if one value is NaN.
namespace Semigroup
  export
  [Max] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = max

namespace Monoid
  export
  [Max] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Max where
      neutral = fill (- 1.0 / 0.0)

---------------------------- other ----------------------------------

||| Cholesky decomposition. Computes the lower triangular matrix `L` from the symmetric, positive
||| semi-definite matrix `X` s.t. `X = L @@ L.T`. Values will be NaN if the input matrix is not
||| positive semi-definite. The remaining matrix components - those not in the lower triangle or
||| diagonal - will always be zero.
export
cholesky : Tensor [S n, S n] F64 -> Tensor [S n, S n] F64
cholesky (MkTensor expr) = triangle Lower $ MkTensor $ Cholesky {leading=[]} {n} expr

infix 9 |\, \|

-- Matrix versions are just vmaps of the Vector versions. Shall we drop the matrix versions?
-- xla::TriangularSolve asks for the matrix version, so might not be worth it
namespace Matrix
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor a) |\ (MkTensor b) = MkTensor $ TriangularSolve {leading=[]} {n=m} {k=n} a b True

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor a) \| (MkTensor b) = MkTensor $ TriangularSolve {leading=[]} {n=m} {k=n} a b False

namespace Vector
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Tensor [m, m] F64 -> Tensor [m] F64 -> Tensor [m] F64
  a |\ b with (b)
    _ | MkTensor {shape=[_]} _ = squeeze (a |\ (expand 1 b))

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m] F64 -> Tensor [m] F64
  a \| b with (b)
    _ | MkTensor {shape=[_]} _ = squeeze (a \| (expand 1 b))

||| Sum the elements along the diagonal of the input. For example,
||| `trace (fromLiteral [[-1, 5], [1, 4]])` is `3`.
export
trace :
  (Primitive.Num dtype, Prelude.Num a) =>
  PrimitiveRW dtype a =>
  Tensor [S n, S n] dtype ->
  Tensor [] dtype
trace x with (x)
  _ | MkTensor {shape=[_, _]} _ = reduce @{Sum} [0, 1] (x * identity)

||| A `Rand a` produces a pseudo-random value of type `a` from a `Tensor [1] U64` state.
||| The state is updated each time a new value is generated.
public export
Rand : Type -> Type
Rand = State (Tensor [1] U64)

inf : Tensor [] F64
inf = fromDouble (1.0 / 0.0)

{-
||| Generate independent and identically distributed (IID) uniform samples bounded element-wise
||| between `bound` and `bound'`.
|||
||| `bound` and `bound'` need not be ordered, and samples will be generated, elementwise, in
||| [min bound bound', max bound bound'). The exception is where the bounds are equal, in which
||| case: if the bounds are finite, samples are generated at the common bound, else samples are NaN.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two uniform samples
||| ```
||| x : Tensor [3] F64
||| x = let key = fromLiteral 2
|||         rng = uniform key (fill 0.0) (fill 1.0)
|||         initialState = fromLiteral [0]
|||      in evalState initialState [| rng * rng |]
||| ```
|||
||| @key Determines the stream of generated samples.
||| @bound A bound of the samples. See full docstring for details.
||| @bound' A bound of the samples. See full docstring for details.
export
uniform :
  {shape : _} ->
  (key : Tensor [] U64) ->
  (bound, bound' : Tensor shape F64) ->
  Rand (Tensor shape F64)
uniform (MkTensor key) bound bound' =
  let minval@(MkTensor minvalExpr) = min bound bound'
      maxval@(MkTensor maxvalExpr) = max bound bound'
   in ST $ \(MkTensor initialState) =>
      let valueState = UniformFloatingPoint key initialState minvalExpr maxvalExpr shape
          value = MkTensor $ GetTupleElement 0 valueState
          -- workaround for XLA bug https://github.com/tensorflow/tensorflow/issues/56663
          -- samples between -inf and 0 should be at -inf, but XLA produces nan
          -- similarly, samples in (inf, inf) should be at inf and respectively for -inf
          inf = broadcast inf
          value = select (minval == - inf && maxval == fill 0) (- inf) value
          value = select (minval == inf && maxval == inf) inf value
          value = select (minval == - inf && maxval == - inf) (- inf) value
       in Id (MkTensor $ GetTupleElement 1 valueState, value)

||| Generate independent and identically distributed (IID) samples from the standard normal
||| distribution.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two normal samples
||| ```
||| x : Tensor [3] F64
||| x = let key = fromLiteral 2
|||         rng = normal key
|||         initialState = fromLiteral [0]
|||      in evalState initialState [| rng * rng |]
||| ```
|||
||| @key Determines the stream of generated samples.
export
normal : {shape : _} -> (key : Tensor [] U64) -> Rand (Tensor shape F64)
normal (MkTensor key) =
  ST $ \(MkTensor initialState) =>
    let valueState = NormalFloatingPoint key initialState shape
     in Id (MkTensor $ GetTupleElement 1 valueState, MkTensor $ GetTupleElement 0 valueState)
-}
