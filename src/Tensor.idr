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
||| This module contains the `Tensor` object, an array of values of arbitrary type, along with a
||| number of functions operating on numeric `Tensor`s.
module Tensor

import Control.Monad.State
import public Data.List
import public Data.List.Elem
import Decidable.Equality
import System.FFI

import Data.Hashable

import Compiler.Computation
import Compiler.Graph
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Constants
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Math
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Matrix
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.ClientLibrary
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.LocalClient
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaComputation
import Compiler.Xla.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.Service.PlatformUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.Xla.TensorFlow.Core.Platform.Status
import Compiler.Xla.TensorFlow.StreamExecutor.Platform
import Literal
import public Primitive
import public Types
import public Util

%hide Xla.Shape

----------------------------- core definitions ----------------------------

||| A `Tensor` is a symbolic value, which may refer to either to a scalar value or array of values,
||| though the runtime representation will likely contain more than its value, and will depend on
||| the specific backend.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (0 shape : Shape) -> (0 dtype : Type) -> Type where
  MkTensor : {shape : _} -> Graph -> Computation XlaOp -> Tensor shape dtype

||| Construct a `Tensor` from `Literal` data.
export
fromLiteral : PrimitiveRW dtype a => {shape : _} -> Literal shape a -> Tensor shape dtype
fromLiteral xs = 
  let graph = FromLiteral {dtype} shape (hashWithSalt defaultSalt xs)
   in MkTensor graph $ cached graph $ do
        MkCachingBuilder builder _ <- get
        literal <- write {dtype} xs
        constantLiteral builder literal

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
toLiteral (MkTensor {shape} _ xs) = unsafePerformIO $ do
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  computation <- build "" xs
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)

||| A string representation of an unevaluated `Tensor`, detailing all enqueued Xla operations.
||| Useful for debugging.
export
Show (Tensor shape dtype) where
  show (MkTensor _ xs) = opToString xs

namespace Bounded
  ||| A type `a` satisfying `Bounded a` has a minimum and maximum value.
  public export
  interface Bounded a where
    min : a
    max : a

||| Finite bounds for numeric tensors.
export
[Finite] Primitive.Num dtype => Bounded (Tensor [] dtype) where
  min = let graph = MinFiniteValue {dtype} in
    MkTensor graph $ cached graph $ do
      MkCachingBuilder builder _ <- get
      minFiniteValue {dtype} builder

  max = let graph = MaxFiniteValue {dtype} in
    MkTensor graph $ cached graph $ do
      MkCachingBuilder builder _ <- get
      maxFiniteValue {dtype} builder

export
Cast (Tensor shape U64) (Tensor shape F64) where
  cast (MkTensor graph xs) =
    let graph' = ConvertElementType {dtype=F64} graph
     in MkTensor graph' $ cached graph' $ do convertElementType {dtype=F64} !xs

----------------------------- structural operations ----------------------------

reshapeWithDefaultOrdering :
  (from, to : Shape) -> Computation XlaOp -> Computation XlaOp
reshapeWithDefaultOrdering from to xs = reshape !xs (range $ length from) to

||| Reshape a `Tensor`. For example, `reshape {to=[2, 1]} (fromLiteral [3, 4])` is
||| `fromLiteral [[3], [4]]`. The output can have a different rank to the input.
export
reshape :
  Primitive dtype =>
  {to : _} ->
  {auto 0 sizesEqual : product from = product to} ->
  Tensor from dtype ->
  Tensor to dtype
reshape (MkTensor {shape=from} graph xs) =
  let graph = Reshape to graph
   in MkTensor graph $ cached graph $ reshapeWithDefaultOrdering from to xs

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
expand axis (MkTensor {shape} graph xs) =
  let to = insertAt axis 1 shape
      graph = Reshape to graph
   in MkTensor graph $ cached graph $ reshapeWithDefaultOrdering shape to xs

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
squeeze (MkTensor {shape=from} graph xs) =
  let graph = Reshape to graph
   in MkTensor graph $ cached graph $ reshapeWithDefaultOrdering from to xs

||| Take a slice from a single `Tensor` axis. For example, for
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
||| `slice 0 1 3 x` is
||| ```
||| y : Tensor [2, 6] S32
||| y = fromLiteral [
|||       [ 6,  7,  8,  9, 10, 11],
|||       [12, 13, 14, 15, 16, 17]
|||     ]
||| ```
||| and `slice 1 0 4 x` to
||| ```
||| z : Tensor [5, 6] S32
||| z = fromLiteral [
|||       [ 0,  1,  2,  3],
|||       [ 6,  7,  8,  9],
|||       [12, 13, 14, 15],
|||       [18, 19, 20, 21],
|||       [24, 25, 26, 27]
|||     ]
||| ```
||| Equal bounds will result in an empty array. For example, `slice 1 2 2 xs` is
||| `fromLiteral [[], [], [], [], []]`.
|||
||| @axis The `Tensor` axis to slice.
||| @from The inclusive lower bound of the slice along the specified `axis`.
||| @to The exclusive upper bound of the slice along the specified `axis`.
export
slice :
  (axis, from, to : Nat) ->
  {auto 0 fromLTEto : from `LTE` to} ->
  {auto 0 axisInBounds : InBounds axis shape} ->
  {auto 0 isWithinAxis : to `LTE` index axis shape} ->
  Primitive dtype =>
  Tensor shape dtype ->
  Tensor (replaceAt axis (to `minus` from) shape) dtype
slice axis from to (MkTensor graph xs) =
  let graph = Slice axis from to graph
      rank = length shape
      start = replicate axis 0 ++ [from] ++ replicate (rank `minus` (S axis)) 0
      stop = replaceAt axis to shape
      strides = replicate rank 1
   in MkTensor graph $ cached graph $ do slice !xs start stop strides

||| Get the `idx`-th element from the specified `axis` of a tensor. For example,
||| `index 0 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is `fromLiteral [3, 4]`, and
||| `index 1 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is `fromLiteral [2, 4, 6]`.
|||
||| @axis The axis to index.
||| @idx Where along the specified `axis` to fetch elements.
export
index :
  Primitive dtype =>
  (axis, idx : Nat) ->
  {auto 0 axisInBounds : InBounds axis shape} ->
  {auto 0 idxInBounds : idx `LT` index axis shape} ->
  Tensor shape dtype ->
  Tensor (deleteAt axis shape) dtype
index axis idx xs with (xs)
  _ | (MkTensor {shape} _ _) =
    let MkTensor graph sliced =
          slice @{lteSuccRight (reflexive {ty=Nat})} axis idx (S idx) xs
        to = deleteAt axis shape
        graph = Reshape to graph
    in MkTensor graph $ cached graph $ reshapeWithDefaultOrdering shape to sliced

||| Split a `Tensor` along a given axis at the specified index. For example,
||| `split 0 2 fromLiteral [[1, 2], [3, 4], [5, 6]]` is
||| `(fromLiteral [[1, 2], [3, 4]], fromLiteral [[5, 6]])`, and
||| `split 1 1 fromLiteral [[1, 2], [3, 4], [5, 6]]` is
||| `(fromLiteral [[1], [3], [5]], fromLiteral [[2], [4], [6]])`.
|||
||| @axis The axis on which to split.
||| @idx The index of the row at which to split the `Tensor`. The elements at the given axis and
|||   index will appear in the right-hand `Tensor`.
export
split :
  forall shape .
  (axis, idx : Nat) ->
  {auto 0 axisInBounds : InBounds axis shape} ->
  {auto 0 idxInBounds : idx + remaining = index axis shape} ->
  Primitive dtype =>
  Tensor shape dtype ->
  (Tensor (replaceAt axis idx shape) dtype, Tensor (replaceAt axis remaining shape) dtype)
split @{_} @{sums} axis idx xs with (xs)
  _ | MkTensor {shape} _ _ =
    let %hint
        isWithinAxis : LTE idx (index axis shape)
        isWithinAxis = rewrite sym sums in lteAddRight idx

        sums' : remaining = minus (index axis shape) idx
        sums' = rewrite sym sums in sym (minusPlus idx)
    in (
          rewrite sym (minusZeroRight idx) in slice axis 0 idx xs,
          rewrite sums' in slice axis idx {isWithinAxis=reflexive {ty=Nat}} (index axis shape) xs
        )

||| Concatenate two `Tensor`s along the specfied `axis`. For example,
||| `concat 0 (fromLiteral [[1, 2], [3, 4]]) (fromLiteral [[5, 6]])` and
||| `concat 1 (fromLiteral [[3], [6]]) fromLiteral ([[4, 5], [7, 8]])` are both
||| `fromLiteral [[1, 2], [3, 4], [5, 6]]`.
export
concat :
  Primitive dtype =>
  (axis : Nat) ->
  Tensor s dtype ->
  Tensor s' dtype ->
  {auto 0 inBounds : (InBounds axis s, InBounds axis s')} ->
  {auto 0 shapesConcatenable : deleteAt axis s = deleteAt axis s'} ->
  Tensor (replaceAt axis (index axis s + index axis s') s) dtype
concat axis (MkTensor graphL l) (MkTensor graphR r) =
  let graph = Concat axis graphL graphR
   in MkTensor graph $ cached graph $ do
        MkCachingBuilder builder _ <- get
        concatInDim builder [!l, !r] (cast axis)

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
diag (MkTensor graph xs) =
  let graph = Diag graph
   in MkTensor graph $ cached graph $ do getMatrixDiagonal !xs

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
triangle tri (MkTensor graph xs) =
  let graph = Triangle (case tri of Upper => False; Lower => True) graph
   in MkTensor graph $ cached graph $ do triangle !xs (case tri of Upper => False; Lower => True)

||| Tranpose a matrix. For example, `(fromLiteral [[1, 2], [3, 4]]).T` is
||| `fromLiteral [[1, 3], [2, 4]]`.
export
(.T) : Primitive dtype => Tensor [m, n] dtype -> Tensor [n, m] dtype
(MkTensor graph xs).T =
  let graph = Transpose graph
   in MkTensor graph $ cached graph $ do transpose !xs [1, 0]

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
identity =
  let graph = Identity {dtype} n
      n = cast n
   in MkTensor graph $ cached graph $ do
        MkCachingBuilder builder _ <- get
        identityMatrix {dtype} builder n n

||| A `DimBroadcastable from to` proves that a dimension of size `from` can be broadcast to a
||| dimension of size `to`.
public export
data DimBroadcastable : (0 from : Nat) -> (0 to : Nat) -> Type where
  ||| Proof that any dimension can be broadcast to itself. For example in shapes `[2, 3]` to
  ||| `[2, 3]`.
  Same : DimBroadcastable x x

  ||| Proof that a dimension of length one can be broadcast to any size. For example in shapes
  ||| `[2, 1]` to `[2, 3]`
  Stack : DimBroadcastable 1 _

  ||| Proof that any dimension can be broadcast to zero. For example in shapes `[2, 3]` to `[2, 0]`.
  Zero : DimBroadcastable _ 0

namespace Broadcastable
  ||| A `Broadcastable from to` constitutes proof that the shape `from` can be broadcast to the
  ||| shape `to`.
  public export
  data Broadcastable : (0 from : Shape) -> (0 to : Shape) -> Type where
    ||| Proof that a shape can be broadcast to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    |||
    ||| Implementation note: we could have used `Broadcast [] []`, which would have resulted in more
    ||| atomic constructors for `Broadcastable`, but the author guesses that this implementation helps
    ||| the type checker avoid applications of `Match`.
    Same : Broadcastable x x

    ||| Proof that a dimension of size `f` can be broadcast to size `t` if these dimensions
    ||| are `DimBroadcastable f t`. For example:
    |||
    ||| [2, 3] to [2, 3]
    ||| [2, 1] to [2, 3]
    ||| [2, 1] to [2, 0]
    Match : forall from, to .
            {auto 0 ranksEq : length from = length to} ->
            {auto 0 dimBroadcastable : DimBroadcastable f t} ->
            Broadcastable from to ->
            Broadcastable (f :: from) (t :: to)

    ||| Proof that broadcasting can add outer dimensions i.e. nesting. For example:
    |||
    ||| [3] to [1, 3]
    ||| [3] to [5, 3]
    Nest : Broadcastable f t -> Broadcastable f (_ :: t)

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
  _ | (MkTensor {shape=from} graph _) =
    let graph = Broadcast to graph
     in case (isElem 0 to, from == to) of
          (Yes _, False) => MkTensor graph $ cached graph $ do
            MkCachingBuilder builder _ <- get
            literal <- allocLiteral {dtype} to
            constantLiteral builder literal
          _ => impl [] to xs

    where

    impl :
      {from, to : _} ->
      (toLeading, toTrailing : List Nat) ->
      {auto prf : Broadcastable from toTrailing} ->
      Tensor from dtype ->
      Tensor to dtype
    impl toLeading _ {prf=Same} (MkTensor _ mkOp) =
      let graph = Broadcast to graph
       in MkTensor graph $ cached graph $
            if (length toLeading == 0) then mkOp else do broadcast !mkOp toLeading
    impl toLeading (th' :: tt') {prf=Match _} (MkTensor _ mkOp) =
      let graph = Broadcast to graph
       in MkTensor graph $ cached graph $ do
            x <- broadcastInDim !mkOp (th' :: tt') (range (length from))
            broadcast x toLeading
    impl toLeading (th' :: tt') {prf=Nest _} xs = impl (toLeading ++ [th']) tt' xs

%hint
export
scalarToAnyOk : (to : Shape) -> Broadcastable [] to
scalarToAnyOk [] = Same
scalarToAnyOk (_ :: xs) = Nest (scalarToAnyOk xs)

||| A `Tensor` where every element has the specified value. For example,
|||
||| ```idris
||| fives : Tensor [2, 3] Int
||| fives = fill 5
||| ```
||| is
||| ```idris
||| fives : Tensor [2, 3] Int
||| fives = fromLiteral [[5, 5, 5], [5, 5, 5]]
||| ```
export
fill : PrimitiveRW dtype ty => {shape : _} -> ty -> Tensor shape dtype
fill = broadcast {shapesOK=scalarToAnyOk shape} . fromLiteral . Scalar

----------------------------- generic operations ----------------------------

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
map f (MkTensor graph xs) =
  let (graph0, p0) = parameter 0 [] "" {dtype=a}
      MkTensor graphf res = f (MkTensor graph0 p0)
      graph = Map graphf [graph]
   in MkTensor graph $ cached graph $ do
        computation <- buildWithSubBuilder "computation" [p0] res
        MkCachingBuilder builder _ <- get
        map builder [!xs] computation (range $ length shape)

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
map2 f (MkTensor graphL l) (MkTensor graphR r) =
  let (graph0, p0) = parameter 0 [] "" {dtype=a}
      (graph1, p1) = parameter 1 [] "" {dtype=b}
      MkTensor graphf res = f (MkTensor graph0 p0) (MkTensor graph1 p1)
      graph = Map graphf [graphL, graphR]
   in MkTensor graph $ cached graph $ do
        computation <- buildWithSubBuilder "computation" [p0, p1] res
        MkCachingBuilder builder _ <- get
        map builder [!l, !r] computation (range $ length shape)

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
  (axis : Nat) ->
  {auto 0 inBounds : InBounds axis shape} ->
  Tensor shape dtype ->
  Tensor (deleteAt axis shape) dtype
reduce axis (MkTensor graph xs) =
  let semigroup : Monoid a -> Semigroup a
      semigroup _ = %search

   in let (graph0, p0) = parameter 0 [] "" {dtype}
          (graph1, p1) = parameter 1 [] "" {dtype}
          MkTensor graphf resf =
            (<+>) @{semigroup reducer} (MkTensor graph0 p0) (MkTensor graph1 p1)
          graph = Reduce graphf axis graph
       in MkTensor graph $ cached graph $ do
            computation <- buildWithSubBuilder "computation" [p0, p1] resf
            let MkTensor _ init = neutral @{reducer}
            reduce !xs !init computation [axis]

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
sort comp dimension (MkTensor graph xs) =
  let (graph0, p0) = parameter 0 [] "" {dtype}
      (graph1, p1) = parameter 1 [] "" {dtype}
      MkTensor graphf fRes = comp (MkTensor graph0 p0) (MkTensor graph1 p1)
      sortedGraph = Sort [graph] graphf dimension False
   in MkTensor sortedGraph $ cached sortedGraph $ do
        comparator <- buildWithSubBuilder "comparator" [p0, p1] fRes
        sort [!xs] comparator dimension False

----------------------------- numeric operations ----------------------------

unaryOp :
  Primitive b => String -> (XlaOp -> Computation XlaOp) -> Tensor shape a -> Tensor shape b
unaryOp fnName xlaOperation (MkTensor graph xs) =
  let graph = ElementwiseUnary fnName graph
   in MkTensor graph $ cached graph $ do xlaOperation !xs

binaryOp :
  Primitive c =>
  String ->
  (XlaOp -> XlaOp -> Computation XlaOp) ->
  Tensor shape a -> Tensor shape b -> Tensor shape c
binaryOp fnName xlaOperation (MkTensor graphL l) (MkTensor graphR r) =
  let graph = ElementwiseBinary fnName graphL graphR
   in MkTensor graph $ cached graph $ do xlaOperation !l !r

||| Element-wise equality. For example, `fromLiteral [1, 2] == fromLiteral [1, 3]` is
||| `fromLiteral [True, False]`.
export
(==) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(==) = binaryOp "(==)" eq

||| Element-wise inequality. For example, `fromLiteral [1, 2] /= fromLiteral [1, 3]` is
||| `fromLiteral [False, True]`.
export
(/=) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(/=) = binaryOp "(/=)" ne

||| Element-wise less than. For example, `fromLiteral [1, 2, 3] < fromLiteral [2, 2, 2]` is
||| `fromLiteral [True, False, False]`.
export
(<) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<) = binaryOp "(<)" lt

||| Element-wise greater than. For example, `fromLiteral [1, 2, 3] > fromLiteral [2, 2, 2]` is
||| `fromLiteral [False, False, True]`.
export
(>) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>) = binaryOp "(>)" gt

||| Element-wise less than or equal. For example, `fromLiteral [1, 2, 3] <= fromLiteral [2, 2, 2]`
||| is `fromLiteral [True, True, False]`.
export
(<=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<=) = binaryOp "(<=)" le

||| Element-wise greater than or equal. For example,
||| `fromLiteral [1, 2, 3] >= fromLiteral [2, 2, 2]` is `fromLiteral [False, True, True]`.
export
(>=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>=) = binaryOp "(>=)" ge

||| Element-wise boolean and. For example,
||| `fromLiteral [True, True, False, False] && fromLiteral [True, False, True, False]` is
||| `fromLiteral [True, False, False, False]`.
export
(&&) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(&&) = binaryOp "(&&)" and

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
(||) = binaryOp "(||)" or

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
not = unaryOp "not" not

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
select (MkTensor gPred pred) (MkTensor gTrue true) (MkTensor gFalse false) =
  let graph = Select gPred gTrue gFalse
   in MkTensor graph $ cached graph $ do select !pred !true !false

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
cond
  (MkTensor graphPred pred)
  onTrue (MkTensor graphTrue true)
  onFalse (MkTensor graphFalse false) =
    let (grapht, pt) = parameter 0 ts "" {dtype=tt}
        (graphf, pf) = parameter 0 fs "" {dtype=ft}
        MkTensor graphOnTrue trueRes = onTrue (MkTensor grapht pt)
        MkTensor graphOnFalse falseRes = onFalse (MkTensor graphf pf)
        graph = Cond graphPred graphOnTrue graphTrue graphOnFalse graphFalse
     in MkTensor graph $ cached graph $ do
          trueComp <- buildWithSubBuilder "truthy computation" [pt] trueRes
          falseComp <- buildWithSubBuilder "falsy computation" [pf] falseRes
          conditional !pred !true trueComp !false falseComp

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
  (MkTensor graphL l) @@ (MkTensor graphR r) =
    let graph = Dot graphL graphR
     in MkTensor graph $ cached graph $ do dot !l !r

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
  (MkTensor graphL l) @@ (MkTensor graphR r) =
    let graph = Dot graphL graphR
     in MkTensor graph $ cached graph $ do dot !l !r

||| Element-wise addition. For example, `fromLiteral [1, 2] + fromLiteral [3, 4]` is
||| `fromLiteral [4, 6]`.
export
(+) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) = binaryOp "(+)" add

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
negate = unaryOp "negate" neg

||| Element-wise subtraction. For example, `fromLiteral [3, 4] - fromLiteral [4, 2]` is
||| `fromLiteral [-1, 2]`.
export
(-) : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) = binaryOp "(-)" sub

||| Element-wise multiplication. For example, `fromLiteral [2, 3] * fromLiteral [4, 5]` is
||| `fromLiteral [8, 15]`.
export
(*) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(*) = binaryOp "(*)" mul

namespace Scalarwise
  ||| Multiplication by a scalar. For example, `fromLiteral 2 * fromLiteral [3, 5]` is
  ||| `fromLiteral [6, 10]`.
  |||
  ||| The RHS is required to be non-scalar simply to avoid ambiguities with element-wise `(*)`.
  export
  (*) : Primitive.Num dtype => Tensor [] dtype -> Tensor (d :: ds) dtype -> Tensor (d :: ds) dtype
  l * r with (r)
    _ | (MkTensor {shape=(d :: ds)} _ _) = (broadcast {shapesOK=scalarToAnyOk (d :: ds)} l) * r

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
(/) = binaryOp "(/)" div

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
    _ | (MkTensor {shape=(d :: ds)} _ _) = l / (broadcast {shapesOK=scalarToAnyOk (d :: ds)} r)

||| The element-wise reciprocal. For example, `recip (fromLiteral [-2, 0, 0.2])`
||| is `fromLiteral [-0.5, nan, 5]`.
export
recip : Tensor shape F64 -> Tensor shape F64
recip = unaryOp "recip" reciprocal

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
(^) = binaryOp "(^)" pow

||| Element-wise absolute value. For example, `abs (fromLiteral [-2, 3])` is
||| `fromLiteral [2, 3]`.
export
abs : Primitive.Abs dtype => Tensor shape dtype -> Tensor shape dtype
abs = unaryOp "abs" abs

||| The element-wise natural exponential. For example, `exp (fromLiteral [-1, 0, 2])` is
||| `fromLiteral [1 / euler, 1, pow euler 2]`.
export
exp : Tensor shape F64 -> Tensor shape F64
exp = unaryOp "exp" exp

||| The element-wise floor function. For example,
||| `floor (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floor : Tensor shape F64 -> Tensor shape F64
floor = unaryOp "floor" floor

||| The element-wise ceiling function. For example,
||| `ceil (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceil : Tensor shape F64 -> Tensor shape F64
ceil = unaryOp "ceil" ceil

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `log (fromLiteral [1 / euler, 1, euler * euler])` is `fromLiteral [-1, 0, 2]`.
export
log : Tensor shape F64 -> Tensor shape F64
log = unaryOp "log" log

||| The element-wise logistic function equivalent to `1 / 1 + exp (-x)`.
export
logistic : Tensor shape F64 -> Tensor shape F64
logistic = unaryOp "logistic" logistic

||| The element-wise sine.
export
sin : Tensor shape F64 -> Tensor shape F64
sin = unaryOp "sin" sin

||| The element-wise cosine.
export
cos : Tensor shape F64 -> Tensor shape F64
cos = unaryOp "cos" cos

||| The element-wise tangent.
export
tan : Tensor shape F64 -> Tensor shape F64
tan = unaryOp "tan" tan

||| The element-wise inverse sine.
export
asin : Tensor shape F64 -> Tensor shape F64
asin = unaryOp "asin" asin

||| The element-wise inverse cosine.
export
acos : Tensor shape F64 -> Tensor shape F64
acos = unaryOp "acos" acos

||| The element-wise inverse tangent.
export
atan : Tensor shape F64 -> Tensor shape F64
atan = unaryOp "atan" atan

||| The element-wise hyperbolic sine.
export
sinh : Tensor shape F64 -> Tensor shape F64
sinh = unaryOp "sinh" sinh

||| The element-wise hyperbolic cosine.
export
cosh : Tensor shape F64 -> Tensor shape F64
cosh = unaryOp "cosh" cosh

||| The element-wise hyperbolic tangent.
export
tanh : Tensor shape F64 -> Tensor shape F64
tanh = unaryOp "tanh" tanh

||| The element-wise inverse hyperbolic sine.
export
asinh : Tensor shape F64 -> Tensor shape F64
asinh = unaryOp "asinh" asinh

||| The element-wise inverse hyperbolic cosine.
export
acosh : Tensor shape F64 -> Tensor shape F64
acosh = unaryOp "acosh" acosh

||| The element-wise inverse hyperbolic tangent.
export
atanh : Tensor shape F64 -> Tensor shape F64
atanh = unaryOp "atanh" atanh

||| An approximation to the element-wise error function.
export
erf : Tensor shape F64 -> Tensor shape F64
erf = unaryOp "erf" erf

||| The element-wise square. For example, `square (fromLiteral [-2, 0, 3])`
||| is `fromLiteral [4, 0, 9]`.
export
square : Tensor shape F64 -> Tensor shape F64
square = unaryOp "square" square

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrt (fromLiteral [0, 9])` is `fromLiteral [0, 3]`.
export
sqrt : Tensor shape F64 -> Tensor shape F64
sqrt = unaryOp "sqrt" sqrt

||| The element-wise minimum of the first argument compared to the second. For example,
||| `min (fromLiteral [-3, -1, 3]) (fromLiteral [-1, 0, 1])` is `fromLiteral [-3, -1, 1]`.
|||
||| **Note:** There is a known issue where sometimes the wrong value is chosen if one value is NaN.
export
min : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
min = binaryOp "min" min

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
max = binaryOp "max" max

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
cholesky (MkTensor graph xs) =
  let graph = Cholesky graph
   in triangle Lower $ MkTensor graph $ cached graph $ do cholesky !xs True

infix 9 |\, \|

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
  (MkTensor graphA a) |\ (MkTensor graphB b) =
    let graph = TriangularSolve True graphA graphB
     in MkTensor graph $ cached graph $ do
          triangularSolve !a !b True True False NoTranspose

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor graphA a) \| (MkTensor graphB b) =
    let graph = TriangularSolve False graphA graphB
     in MkTensor graph $ cached graph $ do
          triangularSolve !a !b True False False NoTranspose

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
    _ | MkTensor {shape=[m]} _ _ = squeeze (a |\ (expand 1 b))

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
    _ | MkTensor {shape=[m]} _ _ = squeeze (a \| (expand 1 b))

||| Sum the elements along the diagonal of the input. For example,
||| `trace (fromLiteral [[-1, 5], [1, 4]])` is `3`.
export
trace :
  (Primitive.Num dtype, Prelude.Num a) =>
  PrimitiveRW dtype a =>
  Tensor [S n, S n] dtype ->
  Tensor [] dtype
trace x with (x)
  _ | MkTensor {shape=[S n, S n]} _ _ = reduce @{Sum} 0 (reduce @{Sum} 1 (x * identity))

||| A `Rand a` produces a pseudo-random value of type `a` from a `Tensor [2] U64` seed.
||| The seed is updated each time a new value is generated.
public export
Rand : Type -> Type
Rand = State (Tensor [2] U64)

||| Generate independent and identically distributed (IID) uniform samples bounded element-wise
||| between `bound` and `bound'` (inclusive). Note `bound` and `bound'` need not be ordered.
|||
||| The generated samples are a deterministic function of the input seed, but may vary between
||| backends and library versions.
export
uniform : {shape : _} -> (bound, bound' : Tensor shape F64) -> Rand (Tensor shape F64)
uniform bound bound' = ST $ \(MkTensor initialStateGraph initialState) =>
  let rngGraph = RngBitGenerator RngThreeFry initialStateGraph shape
      rng = cached rngGraph $ do
        rngBitGenerator RngThreeFry !initialState !(mkShape {dtype=F64} shape)
      newStateGraph = GetTupleElement rngGraph 0
      newState = cached newStateGraph $ do getTupleElement !rng 0
      bitsSampleGraph = GetTupleElement rngGraph 1
      bitsSample : Tensor shape U64 =
        MkTensor bitsSampleGraph $ cached bitsSampleGraph $ do getTupleElement !rng 1
      u64minAsF64 : Tensor [] F64 = cast $ min @{Finite {dtype=U64}}
      u64maxAsF64 : Tensor [] F64 = cast $ max @{Finite {dtype=U64}}
      f64sample = bound + (bound' - bound) *
        (cast bitsSample - broadcast u64minAsF64) / (broadcast $ u64maxAsF64 - u64minAsF64)
   in Id (MkTensor newStateGraph newState, f64sample)
