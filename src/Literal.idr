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
||| specified shape. It is similar to the `Tensor` type, but differs in a number of important ways:
|||
||| * `Literal` offers a convenient syntax for constructing `Literal`s with boolean and numeric
|||   contents. For example, `True`, `1` and `[1, 2, 3]` are all valid `Literal`s. This makes it
|||   useful for constructing `Tensor`s.
||| * `Literal` is *not* accelerated by XLA, so operations on large `Literal`s, and large sequences
|||   of operations on any `Literal`, can be expected to be slower than they would on an equivalent
|||   `Tensor`.
||| * A `Literal` is implemented in pure Idris. As such, it can contain elements of any type, and
|||   implements a number of standard Idris interfaces. This, along with its convenient syntax,
|||   makes it particularly useful for testing operations on `Tensor`s.
module Literal

import Data.Hashable

import public Shape
import Types

||| A scalar or array of values.
public export
data Literal : Shape -> Type -> Type where
  Scalar : a -> Literal [] a
  Nil : Literal (0 :: _) _
  (::) : Literal ds a -> Literal (d :: ds) a -> Literal (S d :: ds) a

export
fromInteger : Integer -> Literal [] Int32
fromInteger = Scalar . cast {to=Int32}

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
  map f (x :: xs) = map f x :: map f xs

functorIdentity : (xs : Literal shape a) -> map Prelude.id xs = xs
functorIdentity (Scalar _) = Refl
functorIdentity [] = Refl
functorIdentity (x :: xs) = cong2 (::) (functorIdentity x) (functorIdentity xs)

functorComposition :
  (xs : Literal shape a) -> (f : a -> b) -> (g : b -> c) -> map (g . f) xs = map g (map f xs)
functorComposition (Scalar _) _ _ = Refl
functorComposition [] _ _ = Refl
functorComposition (x :: xs) f g =
  cong2 (::) (functorComposition x f g) (functorComposition xs f g)

export
{shape : _} -> Applicative (Literal shape) where
  pure x = case shape of
    [] => Scalar x
    (0 :: _) => []
    (S d :: ds) => pure x :: (the (Literal (d :: ds) _) $ pure x)

  (Scalar f) <*> (Scalar x) = Scalar (f x)
  [] <*> [] = []
  (f :: fs) <*> (x :: xs) = (f <*> x) :: (fs <*> xs)

applicativeIdentity : (xs : Literal shape a) -> pure Prelude.id <*> xs = xs
applicativeIdentity (Scalar _) = Refl
applicativeIdentity [] = Refl
applicativeIdentity (x :: xs) = cong2 (::) (applicativeIdentity x) (applicativeIdentity xs)

applicativeHomomorphism :
  (shape : Shape) -> (f : a -> b) -> (x : a) -> pure f <*> pure x = pure {f = Literal shape} (f x)
applicativeHomomorphism [] _ _ = Refl
applicativeHomomorphism (d :: ds) f x = forCons d ds f x
  where
  forCons :
    (d : Nat) ->
    (ds : Shape) ->
    (f : a -> b) ->
    (x : a) ->
    pure f <*> pure x = pure {f = Literal (d :: ds)} (f x)
  forCons 0 _ _ _ = Refl
  forCons (S d) ds f x = cong2 (::) (applicativeHomomorphism ds f x) (forCons d ds f x)

applicativeInterchange :
  (fs : Literal shape (a -> b)) -> (x : a) -> fs <*> pure x = pure ($ x) <*> fs
applicativeInterchange (Scalar _) _ = Refl
applicativeInterchange [] _ = Refl
applicativeInterchange (f :: fs) x =
  cong2 (::) (applicativeInterchange f x) (applicativeInterchange fs x)

applicativeComposition :
  (xs : Literal shape a) ->
  (fs : Literal shape (a -> b)) ->
  (gs : Literal shape (b -> c)) ->
  pure (.) <*> gs <*> fs <*> xs = gs <*> (fs <*> xs)
applicativeComposition (Scalar _) (Scalar _) (Scalar _) = Refl
applicativeComposition [] [] [] = Refl
applicativeComposition (x :: xs) (f :: fs) (g :: gs) =
  cong2 (::) (applicativeComposition x f g) (applicativeComposition xs fs gs)

export
Foldable (Literal shape) where
  foldr f acc (Scalar x) = f x acc
  foldr _ acc [] = acc
  foldr f acc (x :: y) = foldr f (foldr f acc y) x

export
Traversable (Literal shape) where
  traverse f (Scalar x) = [| Scalar (f x) |]
  traverse f [] = pure []
  traverse f (x :: xs) = [| traverse f x :: traverse f xs |]

export
Zippable (Literal shape) where
  zipWith f (Scalar x) (Scalar y) = Scalar (f x y)
  zipWith _ [] [] = []
  zipWith f (x :: xs) (y :: ys) = zipWith f x y :: zipWith f xs ys

  zipWith3 f (Scalar x) (Scalar y) (Scalar z) = Scalar (f x y z)
  zipWith3 _ [] [] [] = []
  zipWith3 f (x :: xs) (y :: ys) (z :: zs) = zipWith3 f x y z :: zipWith3 f xs ys zs

  unzipWith f (Scalar x) = let (x, y) = f x in (Scalar x, Scalar y)
  unzipWith _ [] = ([], [])
  unzipWith f (x :: xs) =
    let (x, y) = unzipWith f x
        (xs, ys) = unzipWith f xs
     in (x :: xs, y :: ys)

  unzipWith3 f (Scalar x) = let (x, y, z) = f x in (Scalar x, Scalar y, Scalar z)
  unzipWith3 _ [] = ([], [], [])
  unzipWith3 f (x :: xs) =
    let (x, y, z) = unzipWith3 f x
        (xs, ys, zs) = unzipWith3 f xs
     in (x :: xs, y :: ys, z :: zs)

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
Eq a => Eq (Literal shape a) where
  x == y = all (zipWith (==) x y)

toVect : Literal (d :: ds) a -> Vect d (Literal ds a)
toVect [] = []
toVect (x :: y) = x :: toVect y

||| Show the `Literal`. The `Scalar` constructor is omitted for brevity.
export
{shape : _} -> Show a => Show (Literal shape a) where
  show = showWithIndent "" where
    showWithIndent : {shape : _} -> String -> Literal shape a -> String
    showWithIndent _ (Scalar x) = show x
    showWithIndent _ [] = "[]"
    showWithIndent {shape=[S _]} _ x = show (toList x)
    showWithIndent {shape=(S d :: dd :: ddd)} indent (x :: xs) =
      let indent = " " ++ indent
          first = showWithIndent indent x
          rest = foldMap (\e => ",\n" ++ indent ++ showWithIndent indent e) (toVect xs)
       in "[" ++ first ++ rest ++ "]"

export
{shape : _} -> Cast (Array shape a) (Literal shape a) where
  cast x with (shape)
    cast x | [] = Scalar x
    cast _ | (0 :: _) = []
    cast (x :: xs) | (S d :: ds) = cast x :: cast xs

export
[toArray] Cast (Literal shape a) (Array shape a) where
  cast (Scalar x) = x
  cast [] = []
  cast (x :: y) = cast @{toArray} x :: cast @{toArray} y

hashWithSaltLiteral : Hashable a => Bits64 -> Literal shape a -> Bits64
hashWithSaltLiteral salt (Scalar x) = hashWithSalt salt x
hashWithSaltLiteral salt [] = hashWithSalt salt (the Bits64 0)
hashWithSaltLiteral salt (x :: xs) = (salt
    `hashWithSalt` the Bits64 1
    `hashWithSaltLiteral` x
  ) `hashWithSaltLiteral` xs

export
Hashable a => Hashable (Literal shape a) where
  hashWithSalt = hashWithSaltLiteral
