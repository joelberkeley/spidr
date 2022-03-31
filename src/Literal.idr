{--
Copyright 2022 Joel Berkeley

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
||| This module contains the `Literal` type, which is a single value or array of values with a
||| specified shape. It is similar to the `Tensor` type, and is useful for constructing `Tensor`s,
||| but it differs in a number of important ways:
|||
||| * A `Literal` can contain arbitrary values.
||| * `Literal` is *not* accelerated by XLA, so operations on large `Literal`s, and large sequences
|||   of operations on any `Literal`, can be expected to be slower than they would on an equivalent
|||   `Tensor`.
||| * `Literal` has a more powerful API, and implements a number of standard Idris interfaces. This
|||   makes it particularly useful for testing the result of `Tensor` operations.
||| * `Literal` offers a convenient syntax for constructing `Literal`s with boolean and numeric
|||   contents. For example, `True`, `1` and `[1, 2, 3]` are all valid `Literal`s.
module Literal

import Data.Hashable

import public Types

prefix 9 #

||| A scalar or array of values.
public export
data Literal : Shape -> Type -> Type where
  Scalar : a -> Literal [] a
  Nil : Literal (0 :: _) _
  (::) : Literal ds a -> Literal (d :: ds) a -> Literal (S d :: ds) a

export
fromInteger : Integer -> Literal [] Int
fromInteger = Scalar . cast {to=Int}

export
fromDouble : Double -> Literal [] Double
fromDouble = Scalar

||| Convenience aliases for scalar boolean literals.
export
True, False : Literal [] Bool
True = Scalar True
False = Scalar False

export
Functor (Literal shape) where
  map f (Scalar x) = Scalar (f x)
  map _ [] = []
  map f (x :: y) = (map f x) :: (map f y)

export
{shape : Shape} -> Applicative (Literal shape) where
  pure x = case shape of
    [] => Scalar x
    (0 :: _) => []
    ((S d) :: ds) => assert_total $ pure x :: pure x

  (Scalar f) <*> (Scalar x) = Scalar (f x)
  _ <*> [] = []
  (fx :: fy) <*> (x :: y) = (fx <*> x) :: (fy <*> y)

export
Foldable (Literal shape) where
  foldr f acc (Scalar x) = f x acc
  foldr _ acc [] = acc
  foldr f acc (x :: y) = foldr f (foldr f acc y) x

||| `True` if no elements are `False`. `all []` is `True`.
export
all : Literal shape Bool -> Bool
all xs = foldr (\x, y => x && y) True xs

export
Num a => Num (Literal [] a) where
  x + y = [| x + y |]
  x * y = [| x * y |]
  fromInteger = Scalar . fromInteger

export
negate : Neg a => Literal shape a -> Literal shape a
negate = map negate

export
{shape : _} -> Eq a => Eq (Literal shape a) where
  x == y = all [| x == y |]

toList' : Literal (d :: ds) a -> List (Literal ds a)
toList' [] = []
toList' (x :: y) = x :: toList' y

showDefaultScalar : Show a => Literal [] a -> String
showDefaultScalar (Scalar x) = show x

showDefaultVector : Show (Literal [] a) => Literal [m] a -> String
showDefaultVector xs = show (toList' xs)

showDefaultMatrix : Show (Literal [] a) => Literal [m, n] a -> String
showDefaultMatrix xs = show (map toList' $ toList' xs)

export
Show (Literal [] Int) where
  show = showDefaultScalar

export
Show (Literal [m] Int) where
  show = showDefaultVector

export
Show (Literal [m, n] Int) where
  show = showDefaultMatrix

export
Show (Literal [] Double) where
  show = showDefaultScalar

export
Show (Literal [m] Double) where
  show = showDefaultVector

export
Show (Literal [m, n] Double) where
  show = showDefaultMatrix

export
Show (Literal [] Bool) where
  show = showDefaultScalar

export
Show (Literal [m] Bool) where
  show = showDefaultVector

export
Show (Literal [m, n] Bool) where
  show = showDefaultMatrix

export
{shape : _} -> Hashable a => Hashable (Literal shape a) where
  hashWithSalt salt (Scalar x) = Data.Hashable.hashWithSalt salt x
  hashWithSalt salt [] = assert_total $ Data.Hashable.hashWithSalt salt 0
  hashWithSalt salt (x :: xs) = assert_total $ salt
    `hashWithSalt` 1
    `hashWithSalt` x
    `hashWithSalt` xs
