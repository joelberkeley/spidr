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
||| This module contains general library utilities.
module Util

import Control.Monad.Identity
import public Control.Monad.Reader
import Data.Contravariant
import public Data.List
import public Data.List.Quantifiers
import public Data.Nat
import public Data.Vect

||| A dependent variant of `flip` where the return type can depend on the input values. `dflip`
||| flips the order of arguments for a function, such that `dflip f x y` is the same as `f y x`.
public export
dflip : {0 c : a -> b -> Type} -> ((x : a) -> (y : b) -> c x y) -> (y : b) -> (x : a) -> c x y
dflip f y x = f x y

||| A `Neq x y` proves `x` is not equal to `y`.
public export 0
Neq : Nat -> Nat -> Type
Neq x y = Either (LT x y) (GT x y)

namespace Vect
  ||| All numbers from `0` to `n - 1` inclusive, in increasing order.
  |||
  ||| @n The (exclusive) limit of the range.
  export
  range : (n : Nat) -> Vect n Nat
  range Z = []
  range (S n) = snoc (range n) n

  ||| Enumerate entries in a vector with their indices. For example, `enumerate [5, 7, 9]`
  ||| is `[(0, 5), (1, 7), (2, 9)]`.
  export
  enumerate : Vect n a -> Vect n (Nat, a)
  enumerate xs =
    let lengthOK = lengthCorrect xs
     in rewrite sym lengthOK in zip (range (length xs)) (rewrite lengthOK in xs)

namespace List
  ||| All numbers from `0` to `n - 1` inclusive, in increasing order.
  |||
  ||| @n The (exclusive) limit of the range.
  export
  range : (n : Nat) -> List Nat
  range n = toList (Vect.range n)

  ||| Enumerate entries in a list with their indices. For example, `enumerate [5, 7, 9]`
  ||| is `[(0, 5), (1, 7), (2, 9)]`.
  export
  enumerate : List a -> List (Nat, a)
  enumerate xs = toList (enumerate (fromList xs))

  ||| `True` if there are no duplicate elements in the list, else `False`.
  |||
  ||| This function has time complexity quadratic in the list length.
  public export
  unique : Eq a => List a -> Bool
  unique [] = True
  unique (x :: xs) = not (elem x xs) && unique xs

  -- for some reason type inference doesn't work on the numbers in this proof if they're
  -- put in the test module
  unique' : HList [
        unique [1] = True
      , unique [0, 1] = True
      , unique [1, 0] = True
      , unique [0, 0] = False
      , unique [0, 0, 1] = False
      , unique [0, 1, 0] = False
      , unique [1, 0, 0] = False
      , unique [1, 1, 0] = False
      , unique [1, 0, 1] = False
      , unique [0, 1, 1] = False
      , unique [1, 2, 3] = True
    ]
  unique' = %search

  namespace All
    ||| Map a constrained function over a list given a list of constraints.
    public export
    map : (f : (x : a) -> {0 ok : p x} -> b) -> (xs : List a) -> {auto 0 allOk : All p xs} -> List b
    map f [] {allOk = []} = []
    map f (x :: xs) {allOk = ok :: _} = f {ok} x :: map f xs

  ||| Index multiple values from a list at once. For example,
  ||| `multiIndex [1, 3] [5, 6, 7, 8]` is `[6, 8]`.
  |||
  ||| @idxs The indices at which to index.
  ||| @xs The list to index.
  public export
  multiIndex : (idxs : List Nat) ->
               (xs : List a) ->
               {auto 0 inBounds : All (flip InBounds xs) idxs} ->
               List a
  multiIndex idxs xs = map (dflip index xs) idxs

  ||| Delete values from a list at specified indices. For example `deleteAt [0, 2] [5, 6, 7, 8]`
  ||| is `[6, 8]`.
  |||
  ||| @idxs The indices of the values to delete.
  ||| @xs The list to delete values from.
  public export
  deleteAt : (idxs : List Nat) ->
             (xs : List a) ->
             {auto 0 inBounds : All (flip InBounds xs) idxs} ->
             List a
  deleteAt idxs xs = impl 0 xs
    where
    impl : Nat -> List a -> List a
    impl _ [] = []
    impl i (x :: xs) = if elem i idxs then impl (S i) xs else x :: impl (S i) xs

  namespace All2
    ||| A binary version of `All` from the standard library.
    public export
    data All2 : (0 p : a -> b -> Type) -> List a -> List b -> Type where
      Nil : All2 p [] []
      (::) : forall xs, ys . p x y -> All2 p xs ys -> All2 p (x :: xs) (y :: ys)

  ||| A `Sorted f xs` proves that for all consecutive elements `x` and `y` in `xs`, `f x y` exists.
  ||| For example, a `Sorted LT xs` proves that all `Nat`s in `xs` appear in increasing numerical
  ||| order.
  public export
  data Sorted : (a -> a -> Type) -> List a -> Type where
    ||| An empty list is sorted.
    SNil : Sorted f []

    ||| Any single element is sorted.
    SOne : Sorted f [x]

    ||| A list is sorted if its tail is sorted and the head is sorted w.r.t. the head of the tail.
    SCons : (y : a) -> f y x -> Sorted f (x :: xs) -> Sorted f (y :: x :: xs)

  ||| If an index is in bounds for a list, it's also in bounds for a longer list
  public export
  inBoundsCons : (xs : List a) -> InBounds k xs -> InBounds k (x :: xs)
  inBoundsCons _ InFirst = InFirst
  inBoundsCons (_ :: ys) (InLater prf) = InLater (inBoundsCons ys prf)

||| Apply a function to the environment of a reader.
export
(>$<) : (env' -> env) -> Reader env a -> Reader env' a
f >$< (MkReaderT g) = MkReaderT (g . f)
