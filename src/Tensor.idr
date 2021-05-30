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
import Data.Nat

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

||| A multidimensional array of a given shape, of elements of a given type.
public export
ArrayLike : Shape -> Type -> Type
ArrayLike [] dtype = dtype
ArrayLike (d :: ds) dtype = Vect d (ArrayLike ds dtype)

||| A `Tensor` contains an array of values, and is differentiated from a nested `Vect` by having
||| its own type and API.
public export
data Tensor : (shape: Shape) -> (dtype: Type) -> Type where
  ||| Construct a `Tensor` from an array.
  MkTensor : ArrayLike shape dtype -> Tensor shape dtype

export
Show (ArrayLike shape dtype) => Show (Tensor shape dtype) where
  show (MkTensor x) = "Tensor " ++ show x

----------------------------- structural operations ----------------------------

infixl 9 ++:  -- todo is this right?
infixl 9 :++  -- todo is this right?

||| Concatenate two Shapes. This function differs from `++` in that the function definition is
||| public and so is fully resolvable at the type level, and from `:++` in the order of the
||| resulting ranks.
public export
(++:) : {0 r, r' : Nat} -> Shape {rank=r} -> Shape {rank=r'} -> Shape {rank=r' + r}
(++:) [] y = rewrite plusZeroRightNeutral r' in y
(++:) {r = S rr} (x :: xs) y = rewrite sym $ plusSuccRightSucc r' rr in x :: (xs ++: y)

||| Concatenate two Shapes. This function differs from `++` in that the function definition is
||| public and so is fully resolvable at the type level, and from `:++` in the order of the
||| resulting ranks.
public export
(:++) : {0 r, r' : Nat} -> Shape {rank=r} -> Shape {rank=r'} -> Shape {rank=r + r'}
(:++) [] y = y
(:++) (x :: xs) y = x :: (xs :++ y)

||| Get the `idx`-th row from a tensor.
|||
||| @idx The row to fetch.
export
index : (idx: Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype
index idx (MkTensor x) = MkTensor $ index idx x

zipWith : {shape : _} -> (a -> b -> c) -> Tensor shape a -> Tensor shape b -> Tensor shape c
zipWith f (MkTensor x) (MkTensor y) = MkTensor (zipWithArray f x y) where
  zipWithArray : {shape': _} ->
                 (a -> b -> c) -> ArrayLike shape' a -> ArrayLike shape' b -> ArrayLike shape' c
  zipWithArray {shape'=[]} f x y = f x y
  zipWithArray {shape'=(d :: ds)} f x y = zipWith (zipWithArray f) x y

||| Tranpose a tensor. For example, `transpose $ MkTensor [[1, 2], [3, 4]]` is
||| `MkTensor [[1, 3], [2, 4]]`.
export
transpose : {n, m : _} -> Tensor [m, n] dtype -> Tensor [n, m] dtype
transpose (MkTensor x) = MkTensor $ transpose x

-- why does this work with a and b but not other names?
-- see http://docs.idris-lang.org/en/latest/tutorial/interfaces.html#functors-and-applicatives
public export
{shape : _} -> Functor (Tensor shape) where
  map f (MkTensor x) = MkTensor (g x) where
    g : {s : _} -> ArrayLike s a -> ArrayLike s b
    g {s = []} y = f y
    g {s = (_ :: _)} ys = map g ys

||| Replicate a tensor over shape `over`.
|||
||| @over The shape over which to replicate the tensor.
export
replicate : {over : Shape} -> Tensor shape dtype -> Tensor (over :++ shape) dtype
replicate (MkTensor x) = MkTensor (f over x) where
  f : (over: Shape) -> ArrayLike shape dtype -> ArrayLike (over :++ shape) dtype
  f [] x' = x'
  f (d :: ds) x' = replicate d (f ds x')

||| Cast the tensor elements to a dtype inferred from the expected type.
export
cast_dtype : Cast dtype dtype' => {shape : _} -> Tensor shape dtype -> Tensor shape dtype'
cast_dtype tensor = map cast tensor

||| Construct a diagonal tensor from the given value, where all off-diagonal elements are zero.
|||
||| @n The length of the tensor rows and columns.
export
diag : Num dtype => (n : Nat) -> dtype -> Tensor [n, n] dtype

-------------------------------- broadcasting -------------------------------

public export
data Broadcastable : (from : Shape) -> (to : Shape) -> Type where
  Same : Broadcastable x x
  -- NOTE : f and t don't need to have the same rank for Widen to be valid, but it keeps Widen
  -- faithful to its name.
  Widen : (f, t : Shape {rank=r}) -> Broadcastable f t -> Broadcastable (1 :: f) (_ :: t)
  Extend : (f, t : Shape {rank=r}) -> Broadcastable f t -> Broadcastable (x :: f) (x :: t)
  Nest : Broadcastable f t -> Broadcastable f (_ :: t)

----------------------------- numeric operations ----------------------------

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

-- here `head` is not the leading dimensions: that would go before
-- each of (head ++: [S n]), (S n :: tail) and (head ++: tail)
||| Matrix multiply two tensors. The tensors are contracted along the last axis of the first tensor
||| and the first axis of the last tensor.
export
(@@) : Num dtype =>
       Tensor (head ++: [S n]) dtype -> Tensor (S n :: tail) dtype -> Tensor (head ++: tail) dtype

||| Element-wise addition.
export
(+) : Num dtype =>
      {l : _} -> Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype

||| Element-wise negation.
export
negate : Neg dtype => Tensor shape dtype -> Tensor shape dtype

||| Element-wise subtraction.
export
(-) : Neg dtype =>
      {l : _} -> Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype 

-- todo do I want a dedicated operator for elementwise multiplication, so that `x * y` is always
--   the mathematical version, and readers can differentiate between that and, say, `x *# y` for
--   elementwise multiplication? Same question for `/`
||| Elementwise multiplication. This reduces to standard tensor multiplication with a scalar for
||| scalar LHS.
export
(*) : Num dtype =>
      {l : _} -> Tensor l dtype -> Tensor r dtype -> {auto prf : Broadcastable l r} -> Tensor r dtype

||| Elementwise floating point division. This reduces to standard tensor division by a scalar for
||| scalar denominator.
export
(/) : Fractional dtype =>
      {l : _} -> Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l dtype

||| The element-wise logarithm.
export
log : Tensor shape Double -> Tensor shape Double

min : Tensor [S _] Double -> Tensor [] Double

---------------------------- other ----------------------------------

||| Element-wise equality.
export
(==) : Eq dtype =
       {l : _} -> Tensor l dtype -> Tensor r dtype -> {auto _ : Broadcastable r l} -> Tensor l Bool

any : Tensor shape Bool -> Tensor [] Bool

all : Tensor shape Bool -> Tensor [] Bool

||| The determinant of a tensor.
export
det : Neg dtype => Tensor [S n, S n] dtype -> Tensor [] dtype

adjugate : Neg dtype => Tensor [S n, S n] dtype -> Tensor [S n, S n] dtype

cholesky : Tensor [S n, S n] dtype => Maybe (Tensor [S n, S n] dtype)

||| The inverse of a matrix.
export
inverse : Tensor [S n, S n] Double -> Maybe $ Tensor [S n, S n] Double

||| The product of all elements along the diagonal of a matrix.
export
trace_product : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
trace_product (MkTensor x) = MkTensor $ product $ diag x
