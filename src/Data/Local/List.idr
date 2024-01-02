{--
Copyright 2024 Joel Berkeley

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
||| This module extends the standard library's Data.List
module Data.Local.List

import public Data.List
import public Data.List.Quantifiers

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

  ||| Concatenate lists of proofs.
  public export
  (++) : All p xs -> All p ys -> All p (xs ++ ys)
  [] ++ pys = pys
  (px :: pxs) ++ pys = px :: (pxs ++ pys)
  
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
