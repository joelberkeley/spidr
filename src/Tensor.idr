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

import Data.SortedMap
import Control.Monad.State
import Data.Hashable
import public Data.List
import public Data.List.Elem
import Decidable.Equality
import System.FFI

import Device
import Error
import Literal
import public Primitive
import public Types
import public Util
import Compiler.XLA
import Compiler.FFI
import Compiler.Graph
import Compiler.TensorFlow.Compiler.XLA.Client.Lib.Math
import Compiler.TensorFlow.Compiler.XLA.Client.Lib.Matrix
import Compiler.TensorFlow.Compiler.XLA.Client.ClientLibrary
import Compiler.TensorFlow.Compiler.XLA.Client.LocalClient
import Compiler.TensorFlow.Compiler.XLA.Client.XlaBuilder
import public Compiler.TensorFlow.Compiler.XLA.Literal
import Compiler.TensorFlow.Compiler.XLA.ShapeUtil

----------------------------- core definitions ----------------------------

||| A `Tensor` is a symbolic value, which may refer to either to a scalar value or array of values,
||| though the runtime representation will likely contain more than its value, and will depend on
||| the specific backend.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (0 shape : Shape) -> (0 dtype : Type) -> Type where
  MkTensor : {shape : _} -> Graph -> ComputationComponent -> Tensor shape dtype

||| Construct a `Tensor` from `Literal` data.
export
fromLiteral : PrimitiveRW dtype a => {shape : _} -> Literal shape a -> Tensor shape dtype
fromLiteral xs = 
  let graph = Leaf "fromLiteral" (hashWithSalt defaultSalt xs) shape (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        lit <- mkLiteral {dtype} xs
        prim__constantLiteral lit graph

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
||| **Note:**
||| * Each call to `toLiteral` will rebuild and execute the graph. Similarly, multiple calls to 
|||   `toLiteral` on different `Tensor`s in a computation will be treated entirely independently.
|||   `toLiteral` does not store intermediate values. This is a known limitation, and may change in
|||   the future.
||| * `toLiteral` performs logging as a side effect. You can disable this by adjusting the
|||   TensorFlow logging level e.g. with `export TF_CPP_MIN_LOG_LEVEL=3`.
|||
||| @device The type of device to execute the graph on.
export
toLiteral :
  {default cpu device : Device} ->
  PrimitiveRW dtype ty =>
  Tensor shape dtype ->
  Literal shape ty
toLiteral (MkTensor {shape} _ xs) = unsafePerformIO $ do
  computation <- build "" xs
  client <- primIO $ prim__getOrCreateLocalClient (platform device) prim__getNullAnyPtr 0
  lit <- prim__executeAndTransfer client computation prim__getNullAnyPtr 0
  pure (toLiteral {dtype} lit)

||| A string representation of an unevaluated `Tensor`, detailing all enqueued XLA operations.
||| Useful for debugging.
export
[XLA] Show (Tensor shape dtype) where
  show (MkTensor _ xs) = unsafePerformIO (prim__opToString xs)

||| A string representation of an unevaluated `Tensor`, detailing all enqueued Idris operations.
||| Useful for debugging.
|||
||| **Note:**
|||   * The layout of the string is not guaranteed. It is intended for humans not machines.
|||   * Idenitifiers used to differentiate `const` values are omitted from the graph for
|||     readability.
export covering
[Graph] Show (Tensor shape dtype) where
  show (MkTensor graph _) = show graph

----------------------------- structural operations ----------------------------

reshapeImpl : (from, to : Shape) -> ComputationComponent -> ComputationComponent
reshapeImpl from to xs = do
  dim_order <- mkIntArray (range (length from))
  cto <- mkIntArray to
  reshaped <- primIO $ prim__reshape !xs dim_order (cast (length from)) cto (cast (length to))
  onCollectAny reshaped XlaOp.delete

||| Reshape a `Tensor`. For example, `reshape {to=[2, 1]} (fromLiteral [3, 4])` is
||| `fromLiteral [[3], [4]]`. The output can have a different rank to the input.
export
reshape : Primitive dtype => {to : _} -> product from = product to
          => Tensor from dtype -> Tensor to dtype
reshape (MkTensor {shape=from} graph xs) =
  let graph = Operation "reshape" [graph] to (typeString {dtype})
   in MkTensor graph $ cached graph $ reshapeImpl from to xs

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified `axis` in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is
||| `fromLiteral [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand : Primitive dtype => (axis : Nat) -> axis `LTE` length shape => Tensor shape dtype
         -> Tensor (insertAt axis 1 shape) dtype
expand axis (MkTensor {shape} graph xs) =
  let graph = Operation "expand" [graph] (insertAt axis 1 shape) (typeString {dtype})
   in MkTensor graph $ cached graph $ reshapeImpl shape (insertAt axis 1 shape) xs

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
squeeze : Primitive dtype => {to : _} -> Squeezable from to => Tensor from dtype -> Tensor to dtype
squeeze (MkTensor {shape=from} graph xs) =
  let graph = Operation "squeeze" [graph] to (typeString {dtype})
   in MkTensor graph $ cached graph $ reshapeImpl from to xs

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
slice : (axis, from, to : Nat) -> from `LTE` to => InBounds axis shape
        => (isWithinAxis : to `LTE` index axis shape) => Primitive dtype
        => Tensor shape dtype -> Tensor (replaceAt axis (to `minus` from) shape) dtype
slice axis from to (MkTensor graph xs) =
  let to_shape = (replaceAt axis (to `minus` from) shape)
      graph = Operation "slice" [graph] to_shape (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        let rank = length shape
        start <- mkIntArray (replicate axis 0 ++ [from] ++ replicate (rank `minus` axis) 0)
        stop <- mkIntArray (replaceAt axis to shape)
        strides <- mkIntArray (the (List Int) $ replicate rank 1)
        sliced <- primIO $ prim__slice !xs start (cast rank) stop (cast rank) strides (cast rank)
        onCollectAny sliced XlaOp.delete

||| Get the `idx`-th element from the specified `axis` of a tensor. For example,
||| `index 0 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is `fromLiteral [3, 4]`, and
||| `index 1 1 $ fromLiteral [[1, 2], [3, 4], [5, 6]]` is `fromLiteral [2, 4, 6]`.
|||
||| @axis The axis to index.
||| @idx Where along the specified `axis` to fetch elements.
export
index : Primitive dtype => (axis, idx : Nat) -> InBounds axis shape => idx `LT` index axis shape
        => Tensor shape dtype -> Tensor (deleteAt axis shape) dtype
index axis idx (MkTensor {shape} graph xs) =
  let graph = Operation "index" [graph] (deleteAt axis shape) (typeString {dtype})
      MkTensor _ sliced : Tensor _ dtype :=
        slice @{lteSuccRight (reflexive {ty=Nat})} axis idx (S idx) (MkTensor {shape} graph xs)
   in MkTensor graph $ cached graph $ reshapeImpl shape (deleteAt axis shape) sliced

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
split : forall shape . (axis, idx : Nat) -> InBounds axis shape
        => idx + remaining = index axis shape => Primitive dtype => Tensor shape dtype
        -> (
            Tensor (replaceAt axis idx shape) dtype,
            Tensor (replaceAt axis remaining shape) dtype
          )
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
concat : Primitive dtype => (axis : Nat) -> Tensor s dtype -> Tensor s' dtype
         -> (InBounds axis s, InBounds axis s') => deleteAt axis s = deleteAt axis s'
         => Tensor (replaceAt axis (index axis s + index axis s') s) dtype
concat axis (MkTensor {shape=s} graphL l) (MkTensor {shape=s'} graphR r) =
  let to_shape = replaceAt axis (index axis s + index axis s') s
      graph = Operation "concat" [graphL, graphR] to_shape (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        operands <- mkXlaOpArray [!l, !r]
        MkXlaBuilder ptr _ <- get
        res <- primIO $ prim__concatInDim ptr operands 2 (cast axis)
        onCollectAny res XlaOp.delete

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
diag (MkTensor {shape=[n, n]} graph xs) =
  let graph = Operation "diag" [graph] [n] (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        xs <- primIO (prim__getMatrixDiagonal !xs)
        onCollectAny xs XlaOp.delete

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
triangle tri (MkTensor {shape=[n, n]} graph xs) =
  let graph = Operation "triangle" [graph] [n, n] (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        op <- primIO $ prim__triangle !xs (case tri of Upper => 0; Lower => 1)
        onCollectAny op XlaOp.delete

||| Tranpose a matrix. For example, `(fromLiteral [[1, 2], [3, 4]]).T` is
||| `fromLiteral [[1, 3], [2, 4]]`.
export
(.T) : Primitive dtype => Tensor [m, n] dtype -> Tensor [n, m] dtype
(MkTensor {shape=[m, n]} graph xs).T =
  let graph = Operation "(.T)" [graph] [n, m] (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        permutations <- mkIntArray $ the (List Int) $ [1, 0]
        op <- primIO $ prim__transpose !xs permutations 2
        onCollectAny op XlaOp.delete

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
  let graph = Leaf "identity" (cast n) [n, n] (typeString {dtype})
      n = cast n
   in MkTensor graph $ cached graph $ do
        MkXlaBuilder ptr _ <- get
        op <- primIO $ prim__identityMatrix ptr (xlaIdentifier {dtype}) n n
        onCollectAny op XlaOp.delete

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
    Match : forall from, to . length from = length to
            => {auto 0 _ : DimBroadcastable f t}
            -> Broadcastable from to
            -> Broadcastable (f :: from) (t :: to)

    ||| Proof that broadcasting can add outer dimensions i.e. nesting. For example:
    |||
    ||| [3] to [1, 3]
    ||| [3] to [5, 3]
    Nest : Broadcastable f t -> Broadcastable f (_ :: t)

empty : Primitive dtype => {shape : _} -> {auto 0 isEmpty : Elem 0 shape} -> Tensor shape dtype
empty = 
  let graph = Leaf "identity" 0 shape (typeString {dtype})
   in MkTensor graph $ cached graph $ do
        xla_shape <- mkShape {dtype} shape
        literal <- primIO $ prim__allocLiteral xla_shape
        literal <- onCollectAny literal Literal.delete
        prim__constantLiteral literal graph

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
broadcast : Primitive dtype => {to : _} -> {auto prf : Broadcastable from to}
            -> Tensor from dtype -> Tensor to dtype
broadcast xs with (xs)
  _ | MkTensor {shape=from} _ _ = case (isElem 0 to, toList from == toList to) of
    (Yes _, False) => empty
    _ => impl [] to xs

    where
    broadcast : List Nat -> ComputationComponent -> ComputationComponent
    broadcast broadcast_sizes xs = do
      broadcast_sizes_ptr <- mkIntArray broadcast_sizes
      op <- primIO (prim__broadcast !xs broadcast_sizes_ptr (cast $ length broadcast_sizes))
      onCollectAny op XlaOp.delete

    broadcastInDim : Shape -> Shape -> ComputationComponent -> ComputationComponent
    broadcastInDim ods bcd xs = do
      ods_ptr <- mkIntArray ods
      bcd_ptr <- mkIntArray bcd
      let len = cast (length ods)
      op <- primIO (prim__broadcastInDim !xs ods_ptr len bcd_ptr len)
      onCollectAny op XlaOp.delete

    impl : {from, to : _} -> (to_leading, to_trailing : List Nat)
      -> {auto prf : Broadcastable from to_trailing} -> Tensor from dtype -> Tensor to dtype
    impl to_leading _ {prf=Same} (MkTensor graph mkOp) =
      let graph = Operation "broadcast" [graph] to (typeString {dtype})
       in MkTensor graph $ if (length to_leading == 0) then mkOp else broadcast to_leading mkOp
    impl to_leading (th' :: tt') {prf=(Match _)} (MkTensor graph mkOp) =
      let graph = Operation "broadcast" [graph] to (typeString {dtype})
       in MkTensor graph $
            broadcast to_leading (broadcastInDim (th' :: tt') (range (length from)) mkOp)
    impl to_leading (th' :: tt') {prf=(Nest _)} xs = impl (to_leading ++ [th']) tt' xs

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
fill = broadcast {prf=scalarToAnyOk shape} . fromLiteral . Scalar

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
map f (MkTensor {shape} graph xs) =
  let graph0 = Leaf "parameter" 0 [] (typeString {dtype=a})
      p0 = cached graph0 $ prim__parameter 0 [] "" {dtype=a}
      MkTensor graphf res = f (MkTensor graph0 p0)
      graph = Operation "map" [graphf, graph] shape (typeString {dtype=b})
   in MkTensor graph $ cached graph $ do
        computation <- buildWithSubBuilder "computation" [p0] res

        operands <- mkXlaOpArray [!xs]
        let rank = length shape
        dimensions <- mkIntArray (range rank)
        MkXlaBuilder ptr _ <- get

        res <- primIO (prim__map
            ptr
            operands 1
            computation
            dimensions (cast rank)
            prim__getNullAnyPtr 0
          )
        onCollectAny res XlaOp.delete

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
map2 : (Primitive a, Primitive b, Primitive c) => (Tensor [] a -> Tensor [] b -> Tensor [] c)
       -> Tensor shape a -> Tensor shape b -> Tensor shape c
map2 f (MkTensor {shape} graphL l) (MkTensor graphR r) =
  let graph0 = Leaf "parameter" 0 [] (typeString {dtype=a})
      graph1 = Leaf "parameter" 1 [] (typeString {dtype=b})
      p0 = cached graph0 $ prim__parameter 0 [] "" {dtype=a}
      p1 = cached graph1 $ prim__parameter 1 [] "" {dtype=b}
      MkTensor graphf res = f (MkTensor graph0 p0) (MkTensor graph1 p1)
      graph = Operation "map2" [graphf, graphL, graphR] shape (typeString {dtype=c})
   in MkTensor graph $ cached graph $ do
        computation <- buildWithSubBuilder "computation" [p0, p1] res

        operands <- mkXlaOpArray [!l, !r]
        let rank = length shape
        dimensions <- mkIntArray (range rank)
        MkXlaBuilder ptr _ <- get

        res <- primIO (prim__map
            ptr
            operands 2
            computation
            dimensions (cast rank)
            prim__getNullAnyPtr 0
          )
        onCollectAny res XlaOp.delete

||| Reduce elements along one `axis` of a `Tensor` according to a specified `reducer` `Monoid`.
||| For example, if `x = fromLiteral [[0, 1, 2], [3, 4, 5]]`, then reduce @{Sum} 0 x` is
||| `fromLiteral [3, 5, 7]` and `reduce @{Sum} 1 x` to `fromLiteral [3, 12]`.
|||
||| @reducer How to reduce elements along the given `axis`.
||| @axis The axis along which to reduce elements.
export
reduce : (reducer : Monoid (Tensor [] dtype)) => Primitive dtype => (axis : Nat) ->
  InBounds axis shape => Tensor shape dtype -> Tensor (deleteAt axis shape) dtype
reduce axis (MkTensor {shape} graph xs) =
  let semigroup : Monoid a -> Semigroup a
      semigroup _ = %search

   in let graph0 = Leaf "parameter" 0 [] (typeString {dtype})
          graph1 = Leaf "parameter" 1 [] (typeString {dtype})
          p0 = cached graph0 $ prim__parameter 0 [] "" {dtype}
          p1 = cached graph1 $ prim__parameter 1 [] "" {dtype}
          MkTensor graphf resf = (<+>) @{semigroup reducer} (MkTensor graph0 p0) (MkTensor graph1 p1)
          graph = Operation "reduce" [graphf, graph] (deleteAt axis shape) (typeString {dtype})
       in MkTensor graph $ cached graph $ do
            computation <- buildWithSubBuilder "computation" [p0, p1] resf
            let MkTensor _ init = neutral @{reducer}
            op <- primIO $ prim__reduce !xs !init computation !(mkIntArray [axis]) 1
            onCollectAny op XlaOp.delete

----------------------------- numeric operations ----------------------------

unaryOp : Primitive b => String -> (GCAnyPtr -> PrimIO AnyPtr) -> Tensor shape a -> Tensor shape b
unaryOp fn_name prim_operator (MkTensor {shape} graph xs) =
  let graph = Operation fn_name [graph] shape (typeString {dtype=b})
   in MkTensor graph $ cached graph $ do
        op <- primIO (prim_operator !xs)
        onCollectAny op XlaOp.delete

binaryOp : Primitive c => String -> (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr)
           -> Tensor shape a -> Tensor shape b -> Tensor shape c
binaryOp fn_name prim_operator (MkTensor {shape} graphL l) (MkTensor graphR r) =
  let graph = Operation fn_name [graphL, graphR] shape (typeString {dtype=c})
   in MkTensor graph $ cached graph $ do
        op <- primIO (prim_operator !l !r)
        onCollectAny op XlaOp.delete

||| Element-wise equality. For example, `fromLiteral [1, 2] == fromLiteral [1, 3]` is
||| `fromLiteral [True, False]`.
export
(==) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(==) = binaryOp "(==)" prim__eq

||| Element-wise inequality. For example, `fromLiteral [1, 2] /= fromLiteral [1, 3]` is
||| `fromLiteral [False, True]`.
export
(/=) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(/=) = binaryOp "(/=)" prim__ne

||| Element-wise less than. For example, `fromLiteral [1, 2, 3] < fromLiteral [2, 2, 2]` is
||| `fromLiteral [True, False, False]`.
export
(<) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<) = binaryOp "(<)" prim__lt

||| Element-wise greater than. For example, `fromLiteral [1, 2, 3] > fromLiteral [2, 2, 2]` is
||| `fromLiteral [False, False, True]`.
export
(>) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>) = binaryOp "(>)" prim__gt

||| Element-wise less than or equal. For example, `fromLiteral [1, 2, 3] <= fromLiteral [2, 2, 2]`
||| is `fromLiteral [True, True, False]`.
export
(<=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<=) = binaryOp "(<=)" prim__le

||| Element-wise greater than or equal. For example,
||| `fromLiteral [1, 2, 3] >= fromLiteral [2, 2, 2]` is `fromLiteral [False, True, True]`.
export
(>=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>=) = binaryOp "(>=)" prim__ge

||| Element-wise boolean and. For example,
||| `fromLiteral [True, True, False, False] && fromLiteral [True, False, True, False]` is
||| `fromLiteral [True, False, False, False]`.
export
(&&) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(&&) = binaryOp "(&&)" prim__and

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
(||) = binaryOp "(||)" prim__or

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
not = unaryOp "not" prim__not

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
select : Primitive dtype => Tensor shape PRED
         -> (onTrue : Tensor shape dtype) -> (onFalse : Tensor shape dtype) -> Tensor shape dtype
select (MkTensor {shape} gPred pred) (MkTensor gTrue true) (MkTensor gFalse false) =
  let graph = Operation "select" [gPred, gTrue, gFalse] shape (typeString {dtype})
  in MkTensor graph $ cached graph $ do
      op <- primIO $ prim__select !pred !true !false
      onCollectAny op XlaOp.delete

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
cond : (Primitive tt, Primitive ft, Primitive dtype) => {shape, ts, fs : _} -> Tensor [] PRED
  -> (onTrue : Tensor ts tt -> Tensor shape dtype) -> Tensor ts tt
  -> (onFalse : Tensor fs ft -> Tensor shape dtype) -> Tensor fs ft
  -> Tensor shape dtype
cond
  (MkTensor graphPred pred)
  onTrue
  (MkTensor graphTrue {shape=tShape} true)
  onFalse
  (MkTensor graphFalse {shape=fShape} false) =
    let grapht = Leaf "parameter" 0 ts (typeString {dtype=tt})
        graphf = Leaf "parameter" 0 fs (typeString {dtype=ft})
        pt = cached grapht $ prim__parameter 0 ts "" {dtype}
        pf = cached graphf $ prim__parameter 0 fs "" {dtype}
        MkTensor graphOnTrue trueRes = onTrue (MkTensor grapht pt)
        MkTensor graphOnFalse falseRes = onFalse (MkTensor graphf pf)
        args = [graphPred, graphOnTrue, graphTrue, graphOnFalse, graphFalse]
        graph = Operation "cond" args shape (typeString {dtype})
     in MkTensor graph $ cached graph $ do
          trueComp <- buildWithSubBuilder "truthy computation" [pt] trueRes
          falseComp <- buildWithSubBuilder "falsy computation" [pf] falseRes
          op <- primIO $ prim__conditional !pred !true trueComp !false falseComp
          onCollectAny op XlaOp.delete

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
    let graph = Operation "(@@)" [graphL, graphR] [] (typeString {dtype})
     in MkTensor graph $ cached graph $ do
          op <- primIO $ prim__dot !l !r
          onCollectAny op XlaOp.delete

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
  (@@) : Primitive dtype => Primitive.Num dtype
         => Tensor [n, S m] dtype -> Tensor (S m :: tl) dtype
         -> length tl `LTE` 1 => Tensor (n :: tl) dtype
  (MkTensor {shape=[n, _]} graphL l) @@ (MkTensor {shape=_ :: tl} graphR r) =
    let graph = Operation "(@@)" [graphL, graphR] (n :: tl) (typeString {dtype})
     in MkTensor graph $ cached graph $ do
          op <- primIO $ prim__dot !l !r
          onCollectAny op XlaOp.delete

||| Element-wise addition. For example, `fromLiteral [1, 2] + fromLiteral [3, 4]` is
||| `fromLiteral [4, 6]`.
export
(+) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) = binaryOp "(+)" prim__add

namespace Semigroup
  export
  [Sum] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (+)

namespace Monoid
  export
  [Sum] {shape : _} -> Prelude.Num a => PrimitiveRW dtype a => Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Sum where
      neutral = fill 0

||| Element-wise negation. For example, `- fromLiteral [1, -2]` is `fromLiteral [-1, 2]`.
export
negate : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype
negate = unaryOp "negate" prim__neg

||| Element-wise subtraction. For example, `fromLiteral [3, 4] - fromLiteral [4, 2]` is
||| `fromLiteral [-1, 2]`.
export
(-) : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) = binaryOp "(-)" prim__sub

||| Element-wise multiplication. For example, `fromLiteral [2, 3] * fromLiteral [4, 5]` is
||| `fromLiteral [8, 15]`.
export
(*) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(*) = binaryOp "(*)" prim__mul

namespace Scalarwise
  ||| Multiplication by a scalar. For example, `fromLiteral 2 * fromLiteral [3, 5]` is
  ||| `fromLiteral [6, 10]`.
  |||
  ||| The RHS is required to be non-scalar simply to avoid ambiguities with element-wise `(*)`.
  export
  (*) : Primitive.Num dtype => Tensor [] dtype -> Tensor (d :: ds) dtype -> Tensor (d :: ds) dtype
  l * r with (r)
    _ | (MkTensor {shape=(d :: ds)} _ _) = (broadcast {prf=scalarToAnyOk (d :: ds)} l) * r

namespace Semigroup
  export
  [Prod] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (*)

namespace Monoid
  export
  [Prod] {shape : _} -> Prelude.Num a => PrimitiveRW dtype a => Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Prod where
      neutral = fill 1

||| Element-wise floating point division. For example, `fromLiteral [2, 3] / fromLiteral [4, 5]` is
||| `fromLiteral [0.5, 0.6]`.
export
(/) : Primitive.Fractional dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(/) = binaryOp "(/)" prim__div

namespace Scalarwise
  ||| Floating point division by a scalar. For example, `fromLiteral [3.4, -5.6] / fromLiteral 2` is
  ||| `fromLiteral [1.7, -2.8]`.
  |||
  ||| The LHS is required to be non-scalar simply to avoid ambiguities with element-wise `(/)`.
  export
  (/) : Primitive.Fractional dtype
        => Tensor (d :: ds) dtype -> Tensor [] dtype -> Tensor (d :: ds) dtype
  l / r with (l)
    _ | (MkTensor {shape=(d :: ds)} _ _) = l / (broadcast {prf=scalarToAnyOk (d :: ds)} r)

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
(^) = binaryOp "(^)" prim__pow

||| Element-wise absolute value. For example, `abs (fromLiteral [-2, 3])` is
||| `fromLiteral [2, 3]`.
export
abs : Primitive.Abs dtype => Tensor shape dtype -> Tensor shape dtype
abs = unaryOp "abs" prim__abs

||| The element-wise natural exponential. For example, `exp (fromLiteral [-1, 0, 2])` is
||| `fromLiteral [1 / euler, 1, pow euler 2]`.
export
exp : Tensor shape F64 -> Tensor shape F64
exp = unaryOp "exp" prim__exp

||| The element-wise floor function. For example,
||| `floor (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floor : Tensor shape F64 -> Tensor shape F64
floor = unaryOp "floor" prim__floor

||| The element-wise ceiling function. For example,
||| `ceil (fromLiteral [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `fromLiteral [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceil : Tensor shape F64 -> Tensor shape F64
ceil = unaryOp "ceil" prim__ceil

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `log (fromLiteral [1 / euler, 1, euler * euler])` is `fromLiteral [-1, 0, 2]`.
export
log : Tensor shape F64 -> Tensor shape F64
log = unaryOp "log" prim__log

||| The element-wise logistic function equivalent to `1 / 1 + exp (-x)`.
export
logistic : Tensor shape F64 -> Tensor shape F64
logistic = unaryOp "logistic" prim__logistic

||| The element-wise sine.
export
sin : Tensor shape F64 -> Tensor shape F64
sin = unaryOp "sin" prim__sin

||| The element-wise cosine.
export
cos : Tensor shape F64 -> Tensor shape F64
cos = unaryOp "cos" prim__cos

||| The element-wise hyperbolic tangent.
export
tanh : Tensor shape F64 -> Tensor shape F64
tanh = unaryOp "tanh" prim__tanh

||| An approximation to the element-wise error function.
export
erf : Tensor shape F64 -> Tensor shape F64
erf = unaryOp "erf" prim__erf

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrt (fromLiteral [0, 9])` is `fromLiteral [0, 3]`.
export
sqrt : Tensor shape F64 -> Tensor shape F64
sqrt = unaryOp "sqrt" prim__sqrt

||| The element-wise minimum of the first argument compared to the second. For example,
||| `min (fromLiteral [-3, -1, 3]) (fromLiteral [-1, 0, 1])` is `fromLiteral [-3, -1, 1]`.
export
min : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
min = binaryOp "min" prim__min

namespace Semigroup
  export
  [Min] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = min

namespace Monoid
  export
  [Min] {shape : _} -> PrimitiveRW dtype Double =>
        Primitive.Fractional dtype => Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Min where
      neutral = fill (1.0 / 0.0)

||| The element-wise maximum of the first argument compared to the second. For example,
||| `max (fromLiteral [-3, -1, 3]) (fromLiteral [-1, 0, 1])` is `fromLiteral [-1, 0, 3]`.
export
max : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
max = binaryOp "max" prim__max

namespace Semigroup
  export
  [Max] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = max

namespace Monoid
  export
  [Max] {shape : _} -> PrimitiveRW dtype Double =>
        Primitive.Fractional dtype => Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Max where
      neutral = fill (- 1.0 / 0.0)

---------------------------- other ----------------------------------

||| Cholesky decomposition. Computes the lower triangular matrix `L` from the symmetric, positive
||| semi-definite matrix `X` s.t. `X = L @@ L.T`.
export
cholesky : Tensor [S n, S n] F64 -> Tensor [S n, S n] F64
cholesky (MkTensor {shape=[S n, _]} graph xs) =
  let graph = Operation "cholesky" [graph] [S n, S n] (typeString {dtype=F64})
   in MkTensor graph $ cached graph $ do
        res <- primIO $ prim__cholesky !xs 1
        onCollectAny res XlaOp.delete

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
  (MkTensor graphA a) |\ (MkTensor {shape=[m, n]} graphB b) =
    let graph = Operation "Matrix.(|\)" [graphA, graphB] [m, n] (typeString {dtype=F64})
     in MkTensor graph $ cached graph $ do
          op <- primIO $ prim__triangularSolve !a !b 1 1 0 1
          onCollectAny op XlaOp.delete

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor graphA a) \| (MkTensor {shape=[m, n]} graphB b) =
    let graph = Operation "Matrix.(\|)" [graphA, graphB] [m, n] (typeString {dtype=F64})
     in MkTensor graph $ cached graph $ do
          op <- primIO $ prim__triangularSolve !a !b 1 0 0 1
          onCollectAny op XlaOp.delete

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
trace : (Primitive.Num dtype, Prelude.Num a) => PrimitiveRW dtype a
        => Tensor [S n, S n] dtype -> Tensor [] dtype
trace x with (x)
  _ | MkTensor {shape=[S n, S n]} _ _ = reduce @{Sum} 0 (reduce @{Sum} 1 (x * identity))
