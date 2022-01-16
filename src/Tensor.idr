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

import Data.Vect
import Data.Vect.Elem
import Decidable.Equality
import System.FFI

import Error
import public Primitive
import public Types
import Util
import XLA.Client.XlaBuilder
import XLA.Client
import XLA

----------------------------- core definitions ----------------------------

||| A `Tensor` is a symbolic value, which may refer to either to a scalar value or array of values,
||| though the runtime representation will likely contain more than its value, and will depend on
||| the specific backend.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (0 shape : Shape {rank}) -> (0 dtype : Type) -> Type where
  Literal : GCAnyPtr -> Tensor shape dtype
  Operand : IO GCAnyPtr -> Tensor shape dtype

||| Construct a `Tensor` from `Array` data.
export
const : Primitive dtype => {shape : _} -> Array shape dtype -> Tensor shape dtype
const xs = Literal $ mkLiteral {rank=length shape} (rewrite lengthCorrect shape in xs)

||| Evaluate a `Tensor`, returning its value as an `Array`.
export
eval : Primitive dtype => {shape : _} -> Tensor shape dtype -> IO $ Array shape dtype
eval (Literal lit) = pure $ toArray lit
eval (Operand op) = do
  -- todo this line seems REALLY wrong. How to get computation from op?
  let computation = build (builder !op)
  client <- primIO prim__localClientOrDie
  lit <- primIO $ prim__executeAndTransfer client computation prim__getNullAnyPtr 0
  lit <- onCollectAny lit delete
  delete computation
  let arr = toArray lit
  pure arr

||| Return a string representation of an unevaluated `Tensor`, detailing all enqueued operations.
||| Useful for debugging.
export
toString : Tensor shape dtype -> IO String
toString (Literal lit) = do
  builder <- primIO (prim__mkXlaBuilder "toString")
  op <- collectXlaOp (constantLiteral builder lit)
  let str = prim__opToString builder op
  delete builder
  pure str

toString (Operand op) = do
  builder <- primIO (prim__mkXlaBuilder "toString")
  let str = prim__opToString builder !op
  delete builder
  pure str

||| Element-wise negation. For example, `- const [1, -2]` is equivalent to `const [-1, 2]`.
export
negate : {shape : _} -> {dtype : _} -> Neg dtype => Tensor shape dtype -> Tensor shape dtype
negate (Literal lit) = Operand $ do
  -- the other option for the Literal case is that we store the literal and pass it to
  -- ExecuteAndTransfer in eval. This would probably involve making an XlaComputation for both
  -- branches. The SO question may answer how to do this.
  builder <- primIO (prim__mkXlaBuilder "negate")
  op <- collectXlaOp $ constantLiteral builder lit
  primIO (prim__neg op) >>= collectXlaOp
negate (Operand op) = Operand $ do
  builder <- primIO (prim__mkXlaBuilder "negate")
  c_shape <- mkIntArray shape
  xla_shape <- primIO $ prim__mkShape 4 c_shape (cast $ length shape)
  param <- collectXlaOp (parameter builder 0 xla_shape "")
  _ <- primIO (prim__neg param) >>= collectXlaOp
  let computation = build builder
  delete xla_shape
  free c_shape

  args <- malloc sizeof_xlaOp
  op <- op
  primIO (prim__setArrayPtr args 0 op)
  op_res <- primIO (prim__call builder computation args 1)  -- todo this seems wrong wrt builder
  free args
  delete builder
  collectXlaOp op_res

----------------------------- structural operations ----------------------------

||| Get the `idx`-th row from a tensor. For example, `index 1 $ const [[1, 2], [3, 4], [5, 6]]`
||| is equivalent to `const [3, 4]`.
|||
||| @idx The row to fetch.
export
index : (idx : Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype

||| Split a `Tensor` along the first axis at the specified index. For example,
||| `split 1 const [[1, 2], [3, 4], [5, 6]]` is equivalent to
||| `(const [[1, 2]], const [[3, 4], [5, 6]])`.
|||
||| @idx The index of the row at which to split the `Tensor`. The row with index `idx` in
|||   the input `Tensor` will appear in the result as the first row in the second `Tensor`.
export
split : (idx : Nat) -> Tensor ((idx + rest) :: tl) dtype
  -> (Tensor (idx :: tl) dtype, Tensor (rest :: tl) dtype)

||| Concatenate two `Tensor`s along their first axis. For example,
||| `concat (const [[1, 2], [3, 4]]) (const [[5, 6]])` is equivalent to
||| `const [[1, 2], [3, 4], [5, 6]]`.
export
concat : Tensor (n :: tl) dtype -> Tensor (m :: tl) dtype -> Tensor ((n + m) :: tl) dtype

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified axis in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ const [[1, 2], [3, 4], [5, 6]]` is equivalent to
||| `const [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand :
  (axis : Fin (S rank)) -> Tensor {rank=rank} shape dtype -> Tensor (insertAt axis 1 shape) dtype

||| Tranpose the last two axes of a tensor. For example, `(const [[1, 2], [3, 4]]).T` is equivalent
||| to `const [[1, 3], [2, 4]]`.
export
(.T) : forall shape, dtype . Tensor shape dtype ->
       let leading = init (init shape)
           m = last (init shape)
           n = last shape
        in Tensor (leading ++ [n, m]) dtype

||| Cast the tensor elements to a new data type.
export
cast_dtype : Cast dtype dtype' => Tensor shape dtype -> Tensor shape dtype'

||| Construct a diagonal tensor from the specified value, where all off-diagonal elements are zero.
||| For example, `the (Tensor [2, 2] Double) (diag 3)` is equivalent to
||| `const [[3.0, 0.0], [0.0, 3.0]]`.
export
diag : Num dtype => Tensor [] dtype -> Tensor [n, n] dtype

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
  data Broadcastable : (0 from : Types.Shape) -> (0 to : Types.Shape) -> Type where
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
    Match : {0 from, to : Shape {rank=r}}
            -> {auto 0 _ : DimBroadcastable f t}
            -> Broadcastable from to
            -> Broadcastable (f :: from) (t :: to)

    ||| Proof that broadcasting can add outer dimensions i.e. nesting. For example:
    |||
    ||| [3] to [1, 3]
    ||| [3] to [5, 3]
    Nest : Broadcastable f t -> Broadcastable f (_ :: t)

empty : Primitive dtype => {shape : Types.Shape} -> {auto isEmpty : Elem 0 shape} -> Tensor shape dtype
empty = const (emptyArray shape) where
  emptyArray : (shape : _) -> {auto isEmpty : Elem Z shape} -> Array shape dtype
  emptyArray {isEmpty = Here} (0 :: _) = []
  emptyArray {isEmpty = (There _)} (d :: ds) = replicate d (emptyArray ds)

||| Broadcast a `Tensor` to a new compatible shape. For example,
|||
||| ```idris
||| x : Tensor [2, 3] Double
||| x = broadcast (const [4, 5, 6])
||| ```
|||
||| is equivalent to
|||
||| ```idris
||| x : Tensor [2, 3] Double
||| x = const [[4, 5, 6], [4, 5, 6]]
||| ```
export
broadcast : Primitive dtype => {from : _} -> {to : _} -> {auto prf : Broadcastable from to}
  -> Tensor from dtype -> Tensor to dtype
-- broadcast xs = case (isElem 0 to, toList from == toList to) of
--   (Yes _, False) => empty
--   _ =>
--     let from_prf = lengthCorrect from
--         to_prf = lengthCorrect to in
--         rewrite sym to_prf in impl {fr=length from} {tr=length to} {tt=length to} []
--           (rewrite to_prf in to) (rewrite from_prf in xs)
--           {prf=rewrite to_prf in rewrite from_prf in prf}

--     where
--     impl : {fr, tr : _} -> {from : Shape {rank=fr}} -> {to : Shape {rank=tr}}
--       -> {tl, tt : _} -> (to_leading : Vect tl Nat) -> (to_trailing : Vect tt Nat)
--       -> {auto prf : Broadcastable from to_trailing} -> Tensor from dtype -> Tensor to dtype
--     impl to_leading _ {prf=Same} (MkTensor raw) =
--       MkTensor $ if (length to_leading == 0) then raw else broadcast raw to_leading
--     impl {fr = (S r)} to_leading (th' :: tt') {prf=(Match _)} (MkTensor raw) =
--       MkTensor $ broadcast (broadcastInDim raw (th' :: tt') (range (S r))) to_leading
--     impl to_leading (th' :: tt') {prf=(Nest _)} xs = impl (to_leading ++ [th']) tt' xs

scalarToAnyOk : (to : Types.Shape) -> Broadcastable [] to
scalarToAnyOk [] = Same
scalarToAnyOk (_ :: xs) = Nest (scalarToAnyOk xs)

namespace Squeezable
  ||| A `Squeezable from to` constitutes proof that the shape `from` can be squeezed to the
  ||| shape `to`. Squeezing is the process of removing any number of dimensions of length one.
  public export
  data Squeezable : (0 from : Types.Shape) -> (0 to : Types.Shape) -> Type where
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
||| x : Tensor [2, 1, 3, 1] Double
||| x = const [[[[4], [5], [6]]], [[[7], [8], [9]]]]
|||
||| y : Tensor [2, 1, 3] Double
||| y = squeeze x
||| ```
|||
||| is equivalent to
|||
||| ```idris
||| y : Tensor [2, 1, 3] Double
||| y = const [[[4, 5, 6]], [[7, 8, 9]]]
||| ```
export
squeeze : {auto 0 _ : Squeezable from to} -> Tensor from dtype -> Tensor to dtype

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
fill : Primitive dtype => {shape : _} -> dtype -> Tensor shape dtype
fill = broadcast {prf=scalarToAnyOk shape} . const

----------------------------- numeric operations ----------------------------

infix 6 ==#, /=#

||| Element-wise equality. For example, `const [1, 2] ==# const [1, 3]` is equivalent to
||| `const [True, False]`.
export
(==#) : Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) ==# (MkTensor r) = MkTensor (eq l r)

||| Element-wise inequality. For example, `const [1, 2] /=# const [1, 3]` is equivalent to
||| `const [False, True]`.
export
(/=#) : Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) /=# (MkTensor r) = MkTensor (ne l r)

infix 6 <#, >#, <=#, >=#

||| Element-wise less than. For example, `const [1, 2, 3] <# const [2, 2, 2]` is equivalent to
||| `const [True, False, False]`.
export
(<#) : Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) <# (MkTensor r) = MkTensor (lt l r)

||| Element-wise greater than. For example, `const [1, 2, 3] ># const [2, 2, 2]` is equivalent to
||| `const [False, False, True]`.
export
(>#) : Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) ># (MkTensor r) = MkTensor (gt l r)

||| Element-wise less than or equal. For example, `const [1, 2, 3] <=# const [2, 2, 2]` is
||| equivalent to `const [True, True, False]`.
export
(<=#) : Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) <=# (MkTensor r) = MkTensor (le l r)

||| Element-wise greater than or equal. For example, `const [1, 2, 3] >=# const [2, 2, 2]` is
||| equivalent to `const [False, True, True]`.
export
(>=#) : Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool
-- (MkTensor l) >=# (MkTensor r) = MkTensor (ge l r)

infixr 5 &&#

||| Element-wise boolean and. For example,
||| `const [True, True, False, False] &&# const [True, False, True, False]` is equivalent to
||| `const [True, False, False, False]`.
export
(&&#) : Tensor shape Bool -> Tensor shape Bool -> Tensor shape Bool
-- (MkTensor l) &&# (MkTensor r) = MkTensor (and l r)

infixr 4 ||#

||| Element-wise boolean or. For example,
||| `const [True, True, False, False] ||# const [True, False, True, False]` is equivalent to
||| `const [True, True, True, False]`.
export
(||#) : Tensor shape Bool -> Tensor shape Bool -> Tensor shape Bool
-- (MkTensor l) ||# (MkTensor r) = MkTensor (or l r)

||| Element-wise boolean negation. For example, `notEach (const [True, False])` is equivalent to
||| `const [False, True]`.
export
notEach : Tensor shape Bool -> Tensor shape Bool
-- notEach (MkTensor raw) = MkTensor (not raw)

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

||| Matrix multiplication. The tensors are contracted along the last axis of the first tensor and
||| the first axis of the last tensor. For example:
|||
||| ```idris
||| x : Tensor [2, 3] Double
||| x = const [[-1, -2, -3], [0, 1, 2]]
|||
||| y : Tensor [3, 1] Double
||| y = const [[4, 0, 5]]
|||
||| z : Tensor [2, 1] Double
||| z = x @@ y
||| ```
|||
||| is equivalent to
|||
||| ```idris
||| z : Tensor [2, 1] Double
||| z = const [-19, 10]
||| ```
export
(@@) : Num dtype => Tensor l dtype -> Tensor (S n :: tail') dtype ->
       {auto 0 _ : last l = S n} -> Tensor (init l ++ tail') dtype

||| Element-wise addition. For example, `const [1, 2] + const [3, 4]` is equivalent to
||| `const [4, 6]`.
export
(+) : Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
-- (MkTensor l) + (MkTensor r) = MkTensor (add l r)

-- ||| Element-wise negation. For example, `- const [1, -2]` is equivalent to `const [-1, 2]`.
-- export
-- negate : Neg dtype => Tensor shape dtype -> Tensor shape dtype
-- negate (MkTensor raw) = MkTensor (neg raw)

||| Element-wise subtraction. For example, `const [3, 4] - const [4, 2]` is equivalent to
||| `const [-1, 2]`.
export
(-) : Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
-- (MkTensor l) - (MkTensor r) = MkTensor (sub l r)

infixl 9 *#, /#

||| Element-wise multiplication. For example, `const [2, 3] *# const [4, 5]` is equivalent to
||| `const [8, 15]`.
export
(*#) : Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
-- (MkTensor l) *# (MkTensor r) = MkTensor (mul l r)

||| Multiplication by a constant. For example, `const 2 * const [3, 5]` is equivalent to
||| `const [6, 10]`.
export
(*) : (Primitive dtype, Num dtype) => Tensor [] dtype -> {shape : _} -> Tensor shape dtype
      -> Tensor shape dtype
-- l * r = (broadcast {prf=scalarToAnyOk shape} l) *# r

||| Element-wise floating point division. For example, `const [2, 3] /# const [4, 5]` is equivalent
||| to `const [0.5, 0.6]`.
export
(/#) : Fractional dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
-- (MkTensor l) /# (MkTensor r) = MkTensor (div l r)

||| Floating point division by a constant. For example, `const [3.4, -5.6] / const 2` is equivalent
||| to `const [1.7, -2.8]`.
export
(/) : (Primitive dtype, Fractional dtype) => {shape : _} ->
      Tensor shape dtype -> Tensor [] dtype -> Tensor shape dtype
-- l / r = l /# (broadcast {prf=scalarToAnyOk shape} r)

||| Element-wise absolute value. For example, `absEach (const [-2, 3])` is equivalent to
||| `const [2, 3]`.
export
absEach : Abs dtype => Tensor shape dtype -> Tensor shape dtype
-- absEach (MkTensor raw) = MkTensor (abs raw)

infixr 9 ^

-- todo we don't support complex yet
||| Each element in `base` raised to the power of the corresponding element in `exponent`.
||| example, `const [2, 25, -9] ^ const [3, -0.5, 0.5]` is equivalent to `const [8, 0.2, 3i]`.
|||
||| Note: The first root is used.
export
(^) : Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype

-- todo
||| The element-wise natural exponential.
export
exp : Tensor shape Double -> Tensor shape Double

-- todo
||| The element-wise natural logarithm.
export
log : Tensor shape Double -> Tensor shape Double

||| Reduce a `Tensor` along the specified `axis` to the smallest element along that axis, removing
||| the axis in the process. For example, `reduce_min 1 $ const [[-1, 5, 10], [4, 5, 6]]` is
||| equivalent to `const [-1, 5, 6]`.
export
reduce_min : Num dtype => (axis : Fin (S r)) -> Tensor {rank=S r} shape dtype ->
  {auto 0 _ : IsSucc $ index axis shape} -> Tensor (deleteAt axis shape) dtype

||| Reduce a `Tensor` along the specified `axis` to the sum of its components, removing the axis in
||| the process. For example, `reduce_sum 1 $ const [[-1, 2, 3], [4, 5, -6]]` is equivalent to
||| `const [3, 7, -3]`.
export
reduce_sum : Num dtype => (axis : Fin (S r)) -> Tensor {rank=S r} shape dtype ->
  {auto 0 _ : IsSucc $ index axis shape} ->  Tensor (deleteAt axis shape) dtype

---------------------------- other ----------------------------------

||| The determinant of a tensor (with respect to the last two axes). For example,
||| `det $ const [[1, 2], [3, 4]]` is equivalent to `const -2`.
export
det : forall shape, dtype . Neg dtype => Tensor shape dtype ->
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

||| Indicates an operation was impossible (at the attempted precision) due to a matrix being
||| singular.
export
data SingularMatrixError = MkSingularMatrixError String

export
Error SingularMatrixError where
  format (MkSingularMatrixError msg) = msg

||| The inverse of a matrix. For example, `inverse $ const [[1, 2], [3, 4]]` is equivalent to
||| `const [[-2, -1], [-1.5, -0.5]]`.
export
inverse : Tensor [S n, S n] Double -> Either SingularMatrixError $ Tensor [S n, S n] Double

||| The product of all elements along the diagonal of a matrix. For example,
||| `trace_product $ const [[2, 3], [4, 5]]` is equivalent to `const 10`.
export
trace_product : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype

||| Sum the elements along the diagonal of the input.
export
trace : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
