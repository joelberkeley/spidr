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

import public Data.List
import public Data.List.Elem
import Decidable.Equality
import System.FFI

import System.FFI

import Error
import public Primitive
import public Types
import public Util
import XLA.Client.Lib.Matrix
import XLA.Client.ClientLibrary
import XLA.Client.LocalClient
import XLA.Client.XlaBuilder
import XLA.FFI
import XLA.Literal
import XLA.ShapeUtil

----------------------------- core definitions ----------------------------

XlaOpFactory : Type
XlaOpFactory = (builder : GCAnyPtr) -> IO GCAnyPtr

||| A `Tensor` is a symbolic value, which may refer to either to a scalar value or array of values,
||| though the runtime representation will likely contain more than its value, and will depend on
||| the specific backend.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (0 shape : Shape) -> (0 dtype : Type) -> Type where
  MkTensor : XlaOpFactory -> Tensor shape dtype

||| Construct a `Tensor` from `Array` data.
export
const : PrimitiveRW dtype ty => {shape : _} -> Array shape ty -> Tensor shape dtype
const xs = MkTensor $ \builder => do
  lit <- mkLiteral {dtype} xs
  onCollectAny (constantLiteral builder lit) XlaOp.delete

||| Evaluate a `Tensor`, returning its value as an `Array`.
export
eval : PrimitiveRW dtype ty => {shape : _} -> Tensor shape dtype -> IO $ Array shape ty
eval (MkTensor mkOp) = do
  builder <- prim__mkXlaBuilder ""
  _ <- mkOp builder
  computation <- prim__build builder
  client <- primIO prim__localClientOrDie
  lit <- prim__executeAndTransfer client computation prim__getNullAnyPtr 0
  pure (toArray {dtype} lit)

||| Return a string representation of an unevaluated `Tensor`, detailing all enqueued operations.
||| Useful for debugging.
export
toString : Tensor shape dtype -> IO String
toString (MkTensor f) = do
  builder <- prim__mkXlaBuilder ""
  pure (prim__opToString builder !(f builder))

----------------------------- structural operations ----------------------------

reshapeImpl : (from, to : Shape) -> GCAnyPtr -> IO GCAnyPtr
reshapeImpl from to op = do
  dim_order <- mkIntArray (range (length from))
  cto <- mkIntArray to
  reshaped <- primIO $ prim__reshape op dim_order (cast (length from)) cto (cast (length to))
  onCollectAny reshaped XlaOp.delete

||| Reshape a `Tensor`. For example, `reshape {to=[2, 1]} (const [3, 4])` is equivalent to
||| `const [[3], [4]]`. The output can have a different rank to the input.
export
reshape : {from, to : _} -> product from = product to => Tensor from dtype -> Tensor to dtype
reshape (MkTensor mkOp) = MkTensor $ \builder => reshapeImpl from to !(mkOp builder)

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified `axis` in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ const [[1, 2], [3, 4], [5, 6]]` is equivalent to
||| `const [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand : (axis : Nat) -> {shape : _} -> axis `LTE` length shape => Tensor shape dtype
         -> Tensor (insertAt axis 1 shape) dtype
expand axis (MkTensor mkOp) = MkTensor $ \builder =>
  reshapeImpl shape (insertAt axis 1 shape) !(mkOp builder)

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
||| x = const [[[[4], [5], [6]]], [[[7], [8], [9]]]]
|||
||| y : Tensor [2, 1, 3] S32
||| y = squeeze x
||| ```
|||
||| is equivalent to
|||
||| ```idris
||| y : Tensor [2, 1, 3] S32
||| y = const [[[4, 5, 6]], [[7, 8, 9]]]
||| ```
export
squeeze : {from, to : _} -> Squeezable from to => Tensor from dtype -> Tensor to dtype
squeeze (MkTensor mkOp) = MkTensor $ \builder => reshapeImpl from to !(mkOp builder)

||| Take a slice from a single `Tensor` axis. For example, for
||| ```
||| x : Tensor [5, 6] S32
||| x = const [
|||       [ 0,  1,  2,  3,  4,  5],
|||       [ 6,  7,  8,  9, 10, 11],
|||       [12, 13, 14, 15, 16, 17],
|||       [18, 19, 20, 21, 22, 23],
|||       [24, 25, 26, 27, 28, 29]
|||     ]
||| ```
||| `slice 0 1 3 x` is equivalent to
||| ```
||| y : Tensor [2, 6] S32
||| y = const [
|||       [ 6,  7,  8,  9, 10, 11],
|||       [12, 13, 14, 15, 16, 17]
|||     ]
||| ```
||| and `slice 1 0 4 x` to
||| ```
||| z : Tensor [5, 6] S32
||| z = const [
|||       [ 0,  1,  2,  3],
|||       [ 6,  7,  8,  9],
|||       [12, 13, 14, 15],
|||       [18, 19, 20, 21],
|||       [24, 25, 26, 27]
|||     ]
||| ```
||| Equal bounds will result in an empty array. For example, `slice 1 2 2 xs` is equivalent to
||| `const [[], [], [], [], []]`.
|||
||| @axis The `Tensor` axis to slice.
||| @from The inclusive lower bound of the slice along the specified `axis`.
||| @to The exclusive upper bound of the slice along the specified `axis`.
export
slice : {shape : _} -> (axis, from, to : Nat)
        -> from `LTE` to => InBounds axis shape => (isWithinAxis : to `LTE` index axis shape)
        => Tensor shape dtype -> Tensor (replaceAt axis (to `minus` from) shape) dtype
slice axis from to (MkTensor mkOp) = MkTensor $ \builder => do
  op <- mkOp builder
  let rank = length shape
  start <- mkIntArray (replicate axis 0 ++ [from] ++ replicate (rank `minus` axis) 0)
  stop <- mkIntArray (replaceAt axis to shape)
  strides <- mkIntArray (replicate rank 1)
  sliced <- primIO $ prim__slice op start (cast rank) stop (cast rank) strides (cast rank)
  onCollectAny sliced XlaOp.delete

||| Get the `idx`-th element from the specified `axis` of a tensor. For example,
||| `index 0 1 $ const [[1, 2], [3, 4], [5, 6]]` is equivalent to `const [3, 4]`, and
||| `index 1 1 $ const [[1, 2], [3, 4], [5, 6]]` is equivalent to `const [2, 4, 6]`.
|||
||| @axis The axis to index.
||| @idx Where along the specified `axis` to fetch elements.
export
index : {shape : _} -> (axis, idx : Nat) -> InBounds axis shape =>
        idx `LT` index axis shape => Tensor shape dtype -> Tensor (deleteAt axis shape) dtype
index axis idx xs@(MkTensor mkOp) =
  let (MkTensor mkSliced) = slice @{lteSuccRight (reflexive {ty=Nat})} axis idx (S idx) xs
   in MkTensor $ \builder => reshapeImpl shape (deleteAt axis shape) !(mkSliced builder)

||| Split a `Tensor` along a given axis at the specified index. For example,
||| `split 0 2 const [[1, 2], [3, 4], [5, 6]]` is equivalent to
||| `(const [[1, 2], [3, 4]], const [[5, 6]])`, and `split 1 1 const [[1, 2], [3, 4], [5, 6]]` to
||| `(const [[1], [3], [5]], const [[2], [4], [6]])`.
|||
||| @axis The axis on which to split.
||| @idx The index of the row at which to split the `Tensor`. The elements at the given axis and
|||   index will appear in the right-hand `Tensor`.
export
split : (axis, idx : Nat) -> {shape : _} -> InBounds axis shape
        => idx + remaining = index axis shape => Tensor shape dtype
        -> (
            Tensor (replaceAt axis idx shape) dtype,
            Tensor (replaceAt axis remaining shape) dtype
          )
split @{_} @{sums} axis idx xs =
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
||| `concat 0 (const [[1, 2], [3, 4]]) (const [[5, 6]])` and
||| `concat 1 (const [[3], [6]]) const ([[4, 5], [7, 8]])` are both equivalent to
||| `const [[1, 2], [3, 4], [5, 6]]`.
export
concat : (axis : Nat) -> Tensor s dtype -> Tensor s' dtype
         -> (InBounds axis s, InBounds axis s') => deleteAt axis s = deleteAt axis s'
         => Tensor (replaceAt axis (index axis s + index axis s') s) dtype
concat axis (MkTensor mkOpL) (MkTensor mkOpR) = MkTensor $ \builder => do
  ops <- mkXlaOpArray [!(mkOpL builder), !(mkOpR builder)]
  res <- primIO $ prim__concatInDim builder ops 2 (cast axis)
  onCollectAny res XlaOp.delete

||| Tranpose the last two axes of a tensor. For example, `(const [[1, 2], [3, 4]]).T` is equivalent
||| to `const [[1, 3], [2, 4]]`.
export
(.T) : forall shape, dtype . NonEmpty shape => NonEmpty (init shape) => Tensor shape dtype ->
       let leading = init (init shape)
           m = last (init shape)
           n = last shape
        in Tensor (leading ++ [n, m]) dtype

||| The identity tensor, with inferred shape and element type. For example,
||| ```
||| x : Tensor [2, 2] S32
||| x = identity
||| ```
||| is equivalent to
||| ```
||| x : Tensor [2, 2] S32
||| x = [[1, 0],
|||      [0, 1]]
||| ```
export
identity : (Primitive.Num dtype, Primitive dtype) => {n : _} -> Tensor [n, n] dtype
identity = MkTensor $ \builder => do
  let n = cast n
  op <- primIO $ prim__identityMatrix builder (xlaIdentifier {dtype}) n n
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

empty : Primitive dtype => {shape : Shape} -> {auto isEmpty : Elem 0 shape} -> Tensor shape dtype
empty = MkTensor $ \builder => do
  xla_shape <- mkShape {dtype} shape
  literal <- primIO $ prim__allocLiteral xla_shape
  literal <- onCollectAny literal Literal.delete
  onCollectAny (constantLiteral builder literal) XlaOp.delete

||| Broadcast a `Tensor` to a new compatible shape. For example,
|||
||| ```idris
||| x : Tensor [2, 3] S32
||| x = broadcast (const [4, 5, 6])
||| ```
|||
||| is equivalent to
|||
||| ```idris
||| x : Tensor [2, 3] S32
||| x = const [[4, 5, 6], [4, 5, 6]]
||| ```
export
broadcast : Primitive dtype => {from : _} -> {to : _} -> {auto prf : Broadcastable from to}
  -> Tensor from dtype -> Tensor to dtype
broadcast xs = case (isElem 0 to, toList from == toList to) of
  (Yes _, False) => empty
  _ => impl [] to xs

    where
    broadcast : List Nat -> XlaOpFactory -> XlaOpFactory
    broadcast broadcast_sizes f builder = do
      broadcast_sizes_ptr <- mkIntArray broadcast_sizes
      op <- primIO (
          prim__broadcast !(f builder) broadcast_sizes_ptr (cast $ length broadcast_sizes)
        )
      onCollectAny op XlaOp.delete

    broadcastInDim : Shape -> Shape -> XlaOpFactory -> XlaOpFactory
    broadcastInDim ods bcd f builder = do
      ods_ptr <- mkIntArray ods
      bcd_ptr <- mkIntArray bcd
      let len = cast (length ods)
      op <- primIO (prim__broadcastInDim !(f builder) ods_ptr len bcd_ptr len)
      onCollectAny op XlaOp.delete

    impl : {from, to : _} -> (to_leading, to_trailing : List Nat)
      -> {auto prf : Broadcastable from to_trailing} -> Tensor from dtype -> Tensor to dtype
    impl to_leading _ {prf=Same} (MkTensor mkOp) =
      MkTensor $ if (length to_leading == 0) then mkOp else broadcast to_leading mkOp
    impl to_leading (th' :: tt') {prf=(Match _)} (MkTensor mkOp) =
      MkTensor $ broadcast to_leading (broadcastInDim (th' :: tt') (range (length from)) mkOp)
    impl to_leading (th' :: tt') {prf=(Nest _)} xs = impl (to_leading ++ [th']) tt' xs

scalarToAnyOk : (to : Shape) -> Broadcastable [] to
scalarToAnyOk [] = Same
scalarToAnyOk (_ :: xs) = Nest (scalarToAnyOk xs)

||| A `Tensor` where every element has the specified value. For example,
|||
||| ```idris
||| fives : Tensor [2, 3] Int
||| fives = fill 5
||| ```
||| is equivalent to
||| ```idris
||| fives : Tensor [2, 3] Int
||| fives = const [[5, 5, 5], [5, 5, 5]]
||| ```
export
fill : PrimitiveRW dtype ty => {shape : _} -> ty -> Tensor shape dtype
fill = broadcast {prf=scalarToAnyOk shape} . const

----------------------------- generic operations ----------------------------

parameter : Int -> String -> (shape : Shape) -> Primitive dtype => IO (Tensor shape dtype)
parameter position name shape = do
  xla_shape <- mkShape {dtype} shape
  pure $ MkTensor $ \builder =>
    onCollectAny (parameter builder position xla_shape name) XlaOp.delete

||| Lift a unary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| recip : Tensor [] F64 -> Tensor [] F64
||| recip = (const 1 /)
||| ```
||| can be lifted to an element-wise reciprocal function as `map recip (const [-2, 0.4])`, which is
||| equivalent to `const [-0.5, 2.5]`.
export
map : (Primitive a, Primitive b) => (Tensor [] a -> Tensor [] b)
      -> {shape : _} -> Tensor shape a -> Tensor shape b
map f (MkTensor mkOp) = MkTensor $ \builder => do
  sub_builder <- prim__createSubBuilder builder "computation"
  (MkTensor mkOp') <- [| f (parameter 0 "" [] {dtype=a}) |]
  _ <- mkOp' sub_builder
  computation <- prim__build sub_builder
  operands <- mkXlaOpArray [!(mkOp builder)]
  let rank = length shape
  dimensions <- mkIntArray (range rank)
  op <- primIO (prim__map
      builder
      operands 1
      computation
      dimensions (cast rank)
      prim__getNullAnyPtr 0
    )
  onCollectAny op XlaOp.delete

||| Lift a binary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| addRecip : Tensor [] F64 -> Tensor [] F64 -> Tensor [] F64
||| addRecip x y = x + const 1 / y
||| ```
||| can be lifted to an element-wise function as
||| `map2 addRecip (const [3.0, -3.0]) (const [-2, 0.4])`, which is equivalent to
||| `const [2.5, -0.5]`.
export
map2 : (Primitive a, Primitive b, Primitive c) => (Tensor [] a -> Tensor [] b -> Tensor [] c)
      -> {shape : _} -> Tensor shape a -> Tensor shape b -> Tensor shape c
map2 f (MkTensor mkOpL) (MkTensor mkOpR) = MkTensor $ \builder => do
  sub_builder <- prim__createSubBuilder builder "computation"
  (MkTensor mkOp') <- [| f (parameter 0 "" [] {dtype=a}) (parameter 1 "" [] {dtype=b}) |]
  _ <- mkOp' sub_builder
  computation <- prim__build sub_builder
  operands <- mkXlaOpArray [!(mkOpL builder), !(mkOpR builder)]
  let rank = length shape
  dimensions <- mkIntArray (range rank)
  op <- primIO (prim__map
      builder
      operands 2
      computation
      dimensions (cast rank)
      prim__getNullAnyPtr 0
    )
  onCollectAny op XlaOp.delete

||| Reduce elements along one `axis` of a `Tensor` according to a specified `reducer` `Monoid`.
||| For example, if `x = const [[0, 1, 2], [3, 4, 5]]`, then reduce @{Sum} 0 x` is equivalent to
||| `const [3, 5, 7]` and `reduce @{Sum} 1 x` to `const [3, 12]`.
|||
||| @reducer How to reduce elements along the given `axis`.
||| @axis The axis along which to reduce elements.
export
reduce : (reducer : Monoid (Tensor [] dtype)) => Primitive dtype => (axis : Nat) ->
  InBounds axis shape => Tensor shape dtype -> Tensor (deleteAt axis shape) dtype
reduce axis (MkTensor mkOp) = MkTensor $ \builder => do
  sub_builder <- prim__createSubBuilder builder "computation"
  (MkTensor mkOp') <- [| (parameter 0 "" [] {dtype}) <+> (parameter 1 "" [] {dtype}) |]
  _ <- mkOp' sub_builder
  let (MkTensor mk_init_value) = neutral @{reducer}
  op <- primIO (prim__reduce
      !(mkOp builder)
      !(mk_init_value builder)
      !(prim__build sub_builder)
      !(mkIntArray [axis])
      1
    )
  onCollectAny op XlaOp.delete

----------------------------- numeric operations ----------------------------

unaryOp : (GCAnyPtr -> PrimIO AnyPtr) -> Tensor shape a -> Tensor shape b
unaryOp prim_operator (MkTensor mkOp) = MkTensor $ \builder => do
  op <- primIO (prim_operator !(mkOp builder))
  onCollectAny op XlaOp.delete

binaryOp : (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr)
           -> Tensor shape a -> Tensor shape b -> Tensor shape c
binaryOp prim_operator (MkTensor mkLeft) (MkTensor mkRight) = MkTensor $ \builder => do
  op <- primIO (prim_operator !(mkLeft builder) !(mkRight builder))
  onCollectAny op XlaOp.delete

infix 6 ==#, /=#

||| Element-wise equality. For example, `const [1, 2] ==# const [1, 3]` is equivalent to
||| `const [True, False]`.
export
(==#) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(==#) = binaryOp prim__eq

||| Element-wise inequality. For example, `const [1, 2] /=# const [1, 3]` is equivalent to
||| `const [False, True]`.
export
(/=#) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(/=#) = binaryOp prim__ne

infix 6 <#, >#, <=#, >=#

||| Element-wise less than. For example, `const [1, 2, 3] <# const [2, 2, 2]` is equivalent to
||| `const [True, False, False]`.
export
(<#) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<#) = binaryOp prim__lt

||| Element-wise greater than. For example, `const [1, 2, 3] ># const [2, 2, 2]` is equivalent to
||| `const [False, False, True]`.
export
(>#) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>#) = binaryOp prim__gt

||| Element-wise less than or equal. For example, `const [1, 2, 3] <=# const [2, 2, 2]` is
||| equivalent to `const [True, True, False]`.
export
(<=#) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<=#) = binaryOp prim__le

||| Element-wise greater than or equal. For example, `const [1, 2, 3] >=# const [2, 2, 2]` is
||| equivalent to `const [False, True, True]`.
export
(>=#) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>=#) = binaryOp prim__ge

infixr 5 &&#

||| Element-wise boolean and. For example,
||| `const [True, True, False, False] &&# const [True, False, True, False]` is equivalent to
||| `const [True, False, False, False]`.
export
(&&#) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(&&#) = binaryOp prim__and

namespace Semigroup
  export
  [All] Semigroup (Tensor shape PRED) where
      (<+>) = (&&#)

namespace Monoid
  export
  [All] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.All where
      neutral = fill True

infixr 4 ||#

||| Element-wise boolean or. For example,
||| `const [True, True, False, False] ||# const [True, False, True, False]` is equivalent to
||| `const [True, True, True, False]`.
export
(||#) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(||#) = binaryOp prim__or

namespace Semigroup
  export
  [Any] Semigroup (Tensor shape PRED) where
      (<+>) = (||#)

namespace Monoid
  export
  [Any] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.Any where
      neutral = fill False

||| Element-wise boolean negation. For example, `notEach (const [True, False])` is equivalent to
||| `const [False, True]`.
export
notEach : Tensor shape PRED -> Tensor shape PRED
notEach = unaryOp prim__not

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

namespace Matrix
  ||| Matrix multiplication. The tensors are contracted along the last axis of the first tensor and
  ||| the first axis of the last tensor. For example:
  |||
  ||| ```idris
  ||| x : Tensor [2, 3] S32
  ||| x = const [[-1, -2, -3], [0, 1, 2]]
  |||
  ||| y : Tensor [3, 1] S32
  ||| y = const [[4, 0, 5]]
  |||
  ||| z : Tensor [2, 1] S32
  ||| z = x @@ y
  ||| ```
  |||
  ||| is equivalent to
  |||
  ||| ```idris
  ||| z : Tensor [2, 1] S32
  ||| z = const [-19, 10]
  ||| ```
  |||
  ||| **WARNING** Not well tested
  export
  (@@) : Primitive.Num dtype => Tensor [n, S m] dtype -> Tensor (S m :: tl) dtype
         -> Tensor (n :: tl) dtype
  (MkTensor mkOpL) @@ (MkTensor mkOpR) = MkTensor $ \builder => do
    op <- primIO $ prim__dot !(mkOpL builder) !(mkOpR builder)
    onCollectAny op XlaOp.delete

namespace Vector
  ||| Vector dot product with a tensor of any rank. The vector dot product is with the first axis of
  ||| the right-hand side tensor. For example:
  |||
  ||| ```idris
  ||| x : Tensor [3] S32
  ||| x = [0, 1, 2]
  |||
  ||| y : Tensor [3, 2] S32
  ||| y = const [[-1, -2], [-3, 0], [1, 2]]
  |||
  ||| z : Tensor [2] S32
  ||| z = x @@ y
  ||| ```
  |||
  ||| is equivalent to
  |||
  ||| ```idris
  ||| z : Tensor [2] S32
  ||| z = const [-1, 4]
  ||| ```
  |||
  ||| **WARNING** Not well tested
  export
  (@@) : Primitive.Num dtype => Tensor [S m] dtype -> Tensor (S m :: tl) dtype -> Tensor tl dtype
  (MkTensor mkOpL) @@ (MkTensor mkOpR) = MkTensor $ \builder => do
    op <- primIO $ prim__dot !(mkOpL builder) !(mkOpR builder)
    onCollectAny op XlaOp.delete

||| Element-wise addition. For example, `const [1, 2] + const [3, 4]` is equivalent to
||| `const [4, 6]`.
export
(+) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) = binaryOp prim__add

namespace Semigroup
  export
  [Sum] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
      (<+>) = (+)

namespace Monoid
  export
  [Sum] {shape : _} -> Prelude.Num a => PrimitiveRW dtype a => Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Sum where
      neutral = fill 0

||| Element-wise negation. For example, `- const [1, -2]` is equivalent to `const [-1, 2]`.
export
negate : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype
negate = unaryOp prim__neg

||| Element-wise subtraction. For example, `const [3, 4] - const [4, 2]` is equivalent to
||| `const [-1, 2]`.
export
(-) : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) = binaryOp prim__sub

infixl 9 *#, /#

||| Element-wise multiplication. For example, `const [2, 3] *# const [4, 5]` is equivalent to
||| `const [8, 15]`.
export
(*#) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(*#) = binaryOp prim__mul

||| Multiplication by a constant. For example, `const 2 * const [3, 5]` is equivalent to
||| `const [6, 10]`.
export
(*) : Primitive dtype => Primitive.Num dtype =>
      Tensor [] dtype -> {shape : _} -> Tensor shape dtype -> Tensor shape dtype
l * r = broadcast {prf=scalarToAnyOk shape} l *# r

namespace Semigroup
  export
  [Prod] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
      (<+>) = (*#)

namespace Monoid
  export
  [Prod] {shape : _} -> Prelude.Num a => PrimitiveRW dtype a => Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Prod where
      neutral = fill 1

||| Element-wise floating point division. For example, `const [2, 3] /# const [4, 5]` is equivalent
||| to `const [0.5, 0.6]`.
export
(/#) : Primitive.Fractional dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(/#) = binaryOp prim__div

||| Floating point division by a constant. For example, `const [3.4, -5.6] / const 2` is equivalent
||| to `const [1.7, -2.8]`.
export
(/) : Primitive dtype => Primitive.Fractional dtype => {shape : _} ->
      Tensor shape dtype -> Tensor [] dtype -> Tensor shape dtype
l / r = l /# broadcast {prf=scalarToAnyOk shape} r

||| Element-wise absolute value. For example, `absEach (const [-2, 3])` is equivalent to
||| `const [2, 3]`.
export
absEach : Primitive.Abs dtype => Tensor shape dtype -> Tensor shape dtype
absEach = unaryOp prim__abs

||| The element-wise natural exponential. For example, `expEach (const [-1, 0, 2])` is equivalent to
||| `const [1 / euler, 1, pow euler 2]`.
export
expEach : Tensor shape F64 -> Tensor shape F64
expEach = unaryOp prim__exp

||| The element-wise floor function. For example,
||| `floorEach (const [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is equivalent to
||| `const [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floorEach : Tensor shape F64 -> Tensor shape F64
floorEach = unaryOp prim__floor

||| The element-wise ceiling function. For example,
||| `ceilEach (const [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is equivalent to
||| `const [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceilEach : Tensor shape F64 -> Tensor shape F64
ceilEach = unaryOp prim__ceil

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `logEach (const [1 / euler, 1, euler * euler])` is equivalent to `const [-1, 0, 2]`.
export
logEach : Tensor shape F64 -> Tensor shape F64
logEach = unaryOp prim__log

||| The element-wise logistic function equivalent to `1 /# 1 + expEach (-x)`.
export
logisticEach : Tensor shape F64 -> Tensor shape F64
logisticEach = unaryOp prim__logistic

||| The element-wise sine.
export
sinEach : Tensor shape F64 -> Tensor shape F64
sinEach = unaryOp prim__sin

||| The element-wise cosine.
export
cosEach : Tensor shape F64 -> Tensor shape F64
cosEach = unaryOp prim__cos

||| The element-wise hyperbolic tangent.
export
tanhEach : Tensor shape F64 -> Tensor shape F64
tanhEach = unaryOp prim__tanh

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrtEach (const [0, 9])` is equivalent to `const [0, 3]`.
export
sqrtEach : Tensor shape F64 -> Tensor shape F64
sqrtEach = unaryOp prim__sqrt

infixr 9 ^

||| Each element in `base` raised to the power of the corresponding element in `exponent`.
||| example, `const [2, 25, -9] ^ const [3, -0.5, 0.5]` is equivalent to `const [8, 0.2, 3i]`.
|||
||| Note: The first root is used.
export
(^) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype

||| The element-wise minimum of the first argument compared to the second. For example,
||| `minEach (const [-3, -1, 3]) (const [-1, 0, 1])` is equivalent to `const [-3, -1, 1]`.
export
minEach : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
minEach = binaryOp prim__min

namespace Semigroup
  export
  [Min] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = minEach

namespace Monoid
  export
  [Min] {shape : _} -> PrimitiveRW dtype Double =>
        Primitive.Fractional dtype => Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Min where
      neutral = fill (1.0 / 0.0)

||| The element-wise maximum of the first argument compared to the second. For example,
||| `maxEach (const [-3, -1, 3]) (const [-1, 0, 1])` is equivalent to `const [-1, 0, 3]`.
export
maxEach : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
maxEach = binaryOp prim__max

namespace Semigroup
  export
  [Max] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = maxEach

namespace Monoid
  export
  [Max] {shape : _} -> PrimitiveRW dtype Double =>
        Primitive.Fractional dtype => Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Max where
      neutral = fill (- 1.0 / 0.0)

---------------------------- other ----------------------------------

||| The determinant of a tensor (with respect to the last two axes). For example,
||| `det $ const [[1, 2], [3, 4]]` is equivalent to `const -2`.
export
det : forall shape, dtype . Primitive.Neg dtype => NonEmpty shape => NonEmpty (init shape)
      => Tensor shape dtype ->
      let leading = init (init shape)
          m = last (init shape)
          n = last shape
       in {auto 0 isSquare : m = n} -> {auto 0 nonEmpty : IsSucc m} -> Tensor leading dtype

||| Cholesky decomposition. Finds the lower triangular matrix `L` from `X` s.t. `X = L @@ L.T`.
export
cholesky : Tensor [S n, S n] dtype -> Tensor [S n, S n] dtype

infix 9 \\

||| Find `Y` from `A` and `X` s.t. `X = AY` where `A` is a lower triangular matrix.
export
(\\) : Tensor [n, n] dtype -> Tensor (n :: tl) dtype -> Tensor (n :: tl) dtype

||| Sum the elements along the diagonal of the input.
export
trace : Primitive.Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
