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

import public Data.Vect
import Data.Nat
import Poplar
import Util

----------------------------- core definitions ----------------------------

||| Describes the shape of a `Tensor`. For example, a `Tensor` of `Double`s with contents
||| `[[0, 1, 2], [3, 4, 5]]` has two elements in its outer-most axis, and each of those elements
||| has three `Double`s in it, so this has shape [2, 3]. A `Tensor` can have axes of zero length,
||| though the shape cannot be unambiguously inferred by visualising it. For example, `[[], []]`
||| can have shape [2, 0], [2, 0, 5] or etc. A scalar `Tensor` has shape `[]`.
|||
||| The rank is the number of elements in the shape, or equivalently the number of axes.
public export
Shape : {rank: Nat} -> Type
Shape {rank} = Vect rank Nat

||| A `ScalarLike` is any Idris type that can be represented as a scalar `Tensor`. For a Poplar
||| backend, these types must be convertible to types supported by the IPU.
export
interface ScalarLike ty where
  archType : ArchType

export
ScalarLike Double where
  archType = ?F64

export
ScalarLike Int where
  archType = U32

export
ScalarLike Integer where
  archType = U64

export
ScalarLike Nat where
  archType = I64

export
ScalarLike Bool where
  archType = BOOL

||| A multidimensional array of a given shape, of elements of a given type.
public export 0
Array : {0 dtype : Type} -> ScalarLike dtype => Shape -> Type
Array {dtype} [] = dtype
Array {dtype} (d :: ds) = Vect d (Array ds {dtype=dtype})

||| A `Tensor` is either a scalar value or array of values.
export
data Tensor : (shape : Shape) -> (dtype : Type) -> Type where
  MkTensor : ScalarLike dtype => Array shape {dtype=dtype} -> Tensor shape dtype

||| Construct a `Tensor` from `Array` data.
export
const : ScalarLike dtype => Array shape {dtype=dtype} -> Tensor shape dtype
const = MkTensor

||| Represents a mutable tensor. That is, a tensor that can be modified in-place.
|||
||| We can do this in Idris with linearity. Linearity is offered by quantitative type theory*, which
||| allows us to guarantee that a value is used at run time either never (erased), once (linear), or
||| more. In-place mutation traditionally suffers from the problem that you have to reason about
||| what state a value is in in a series of computations: whether it has been mutated and how. For
||| example, in the following pseudo-code,
|||
||| > a = 1
||| > a += 1
||| > b = f(a)
|||
||| We have to be aware of whether `a` was modified between its initialization and its use in the
||| calculation of `b`. This problem is solved by simply defining a new variable, as
|||
||| > a = 1
||| > a' = a + 1
||| > b = f(a')
|||
||| but this doesn't provide the same performance benefits of in-place mutation. The conundrum is
||| (at least largely) solved with linear types, because you can require that the action of mutating
||| a value "uses it up" such that the previous reference to it cannot be used any more. In the
||| first example, the mutation `a += 1` would use up `a` and we wouldn't be able to use it in the
||| construction of `b`, so the problem no longer exists.
|||
||| In order to ensure `Variable` is only used as a linear type, it is accessible only via the
||| function `var`.
|||
||| *See http://www.type-driven.org.uk/edwinb
export
data Variable : (shape : Shape) -> (dtype : Type) -> Type where
  MkVariable : ScalarLike dtype => Array shape {dtype=dtype} -> Variable shape dtype

||| Provides access to a linear `Variable` type with contents `arr`. For example:
|||
||| > addOne : (1 v : Variable [] Double) -> Variable [] Double
||| > addOne v = v += const {shape=[]} 1
||| >
||| > three : Tensor [] Double
||| > three = var 2.0 $ \v => freeze $ addOne v
|||
||| @arr The initial contents of the `Variable`.
||| @f A function which uses the `Variable`. The return value of `f` is returned by `var`.
var : ScalarLike dtype =>
      Array shape {dtype=dtype} -> (1 f : (1 v : Variable shape dtype) -> a) -> a
var arr f = f (MkVariable arr)

||| Convert a `Variable` to a `Tensor`.
freeze : (1 _ : Variable shape dtype) -> Tensor shape dtype

----------------------------- structural operations ----------------------------

||| Get the `idx`-th row from a tensor.
|||
||| @idx The row to fetch.
export
index : (idx : Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype

||| Tranpose a tensor. For example, `transpose $ const [[1, 2], [3, 4]]` is
||| `const [[1, 3], [2, 4]]`.
export
transpose : Tensor [m, n] dtype -> Tensor [n, m] dtype

||| A `Tensor` where every element has the specified value.
export
fill : dtype -> Tensor shape dtype

||| Replicate a tensor over shape `over`.
|||
||| @over The shape over which to replicate the tensor.
export
replicate : Tensor shape dtype -> Tensor (over ++ shape) dtype

||| Cast the tensor elements to a dtype inferred from the expected type.
export
cast_dtype : Cast dtype dtype' => {shape : _} -> Tensor shape dtype -> Tensor shape dtype'

||| Construct a diagonal tensor from the specified value, where all off-diagonal elements are zero.
export
diag : Num dtype => dtype -> Tensor [n, n] dtype

namespace ns_broadcastable
  ||| A `Broadcastable from to` constitutes proof that the shape `from` can be broadcasted to the
  ||| shape `to`.
  public export
  data Broadcastable : (from : Shape) -> (to : Shape) -> Type where
    ||| Proof that a shape can be broadcast to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    |||
    ||| Implementation note: we could have used `Broadcast [] []`, which would have been more atomic
    ||| wrt. the other constructors, but the author guesses that this implementation helps the type
    ||| checker avoid applications of `Extend`.
    Same : Broadcastable x x
  
    ||| Proof that any dimension with size one can be stacked to any size. For example:
    |||
    ||| [1, 3] to [5, 3]
    ||| [3, 1, 2] to [3, 5, 2]
    Stack : Broadcastable f (1 :: t) -> Broadcastable f (S (S _) :: t)
  
    ||| Proof that any dimension can be broadcast to itself. For example:
    |||
    ||| [2, ...] to [2, ...], assuming the ellipses are broadcast-compatible.
    |||
    ||| Implementation note: the ranks must be equal so that the dimensions are added along the same
    ||| axes.
    Extend : (f, t : Shape {rank=r}) -> Broadcastable f t -> Broadcastable (x :: f) (x :: t)
  
    ||| Proof that broadcasting can add outer dimensions i.e. nesting.
    |||
    ||| [3] to [1, 3]
    Nest : Broadcastable f t -> Broadcastable f (1 :: t)

||| Broadcast a `Tensor` to a new compatible shape.
export
broadcast : {auto prf : Broadcastable from to} -> Tensor from dtype -> Tensor to dtype

namespace ns_squeezable
  ||| A `Squeezable from to` constitutes proof that the shape `from` can be squeezed to the
  ||| shape `to`. Squeezing is the process of removing any number of dimensions of length one.
  public export
  data Squeezable : (from : Shape) -> (to : Shape) -> Type where
    ||| Proof that a shape can be squeezed to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    Same : Squeezable x x

    ||| Proof that any dimensions can be preserved in the process of squeezing. For example:
    |||
    ||| ...
    Extend : Squeezable from to -> Squeezable (x :: from) (x :: to)

    ||| Proof that any dimensions of length one can be squeezed out. For example:
    |||
    ||| [1, 3, 1, 1, 4] to [3, 4]
    Nest : Squeezable from to -> Squeezable (1 :: from) to

||| Remove dimensions of length one from a `Tensor` such that it has the desired shape.
export
squeeze : {auto _ : Squeezable from to} -> Tensor from dtype -> Tensor to dtype

----------------------------- numeric operations ----------------------------

||| Element-wise equality.
export
(==) : Eq dtype => Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

||| Element-wise inequality.
export
(/=) : Eq dtype => Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

||| Element-wise less than.
export
(<) : Ord dtype => Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

||| Element-wise greater than.
export
(>) : Ord dtype => Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

||| Element-wise less than or equal.
export
(<=) : Ord dtype =>
       Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

||| Element-wise greater than or equal.
export
(>=) : Ord dtype =>
       Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

||| Matrix multiplication. The tensors are contracted along the last axis of the first tensor and
||| the first axis of the last tensor.
export
(@@) : Num dtype => Tensor l dtype -> Tensor (S n :: tail) dtype ->
       {auto prf : last l = S n} -> Tensor (init l ++ tail) dtype

||| Element-wise addition.
export
(+) : Num dtype =>
      Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype

||| Element-wise negation.
export
negate : Neg dtype => Tensor shape dtype -> Tensor shape dtype

||| Element-wise subtraction.
export
(-) : Neg dtype =>
      Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype 

||| Elementwise multiplication. This reduces to standard tensor multiplication with a scalar for
||| scalar LHS.
export
(*) : Num dtype =>
      Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable l r} -> Tensor r dtype

||| Elementwise floating point division. This reduces to standard tensor division by a scalar for
||| scalar denominator.
export
(/) : Fractional dtype =>
      Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype

infix 8 +=
infix 8 -=
infix 8 *=
infix 8 /=

||| Element-wise in-place addition. It is in-place in the sense that the value in memory is mutated
||| in-place. However, you must still use the result to get the updated value. For example:
|||
||| > addOne : (1 v : Variable [] Double) -> Variable [] Double
||| > addOne v = v += 1
export
(+=) : Num dtype =>
  (1 v : Variable l dtype) -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Variable l dtype

||| Element-wise in-place subtraction. See `(+=)` for details.
export
(-=) : Neg dtype =>
  (1 v : Variable l dtype) -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Variable l dtype

||| Element-wise in-place multiplication. See `(+=)` for details.
export
(*=) : Num dtype =>
  (1 v : Variable l dtype) -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Variable l dtype

||| Element-wise in-place division. See `(+=)` for details.
export
(//=) : Fractional dtype =>
  (1 v : Variable l dtype) -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Variable l dtype

||| The element-wise logarithm.
export
log : Tensor shape Double -> Tensor shape Double

export
reduce_min : Tensor (S _ :: tail) Double -> Tensor tail Double

---------------------------- other ----------------------------------

any : Tensor shape Bool -> Tensor [] Bool

all : Tensor shape Bool -> Tensor [] Bool

||| The determinant of a tensor.
export
det : Neg dtype => Tensor [S n, S n] dtype -> Tensor [] dtype

||| Indicates a Cholesky decomposition was impossible (at the attempted precision).
export
data CholeskyError = MkCholeskyError String

export
Error CholeskyError where
  format (MkCholeskyError msg) = msg

export
cholesky : Tensor [S n, S n] dtype => Either CholeskyError $ Tensor [S n, S n] dtype

||| Indicates an operation was impossible (at the attempted precision) due to a matrix being
||| singular.
export
data SingularMatrixError = MkSingularMatrixError String

export
Error SingularMatrixError where
  format (MkSingularMatrixError msg) = msg

||| The inverse of a matrix.
export
inverse : Tensor [S n, S n] Double -> Either SingularMatrixError $ Tensor [S n, S n] Double

||| The product of all elements along the diagonal of a matrix.
export
trace_product : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
