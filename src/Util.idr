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

  namespace All
    ||| Map a constrained function over a list given a list of constraints.
    public export
    map : (f : (x : a) -> {0 ok : p x} -> b) -> (xs : List a) -> {auto 0 allOk : All p xs} -> List b
    map f [] {allOk = []} = []
    map f (x :: xs) {allOk = ok :: _} = f {ok} x :: map f xs

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
