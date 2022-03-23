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
import public Data.Nat
import public Data.Vect

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
  enumerate : Vect n ty -> Vect n (Nat, ty)
  enumerate xs = rewrite sym $ lengthCorrect xs in
    zip (range (length xs)) (rewrite lengthCorrect xs in xs)

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
  enumerate : List ty -> List (Nat, ty)
  enumerate xs = toList (enumerate (fromList xs))

  ||| Insert a value in a list. For example, `insertAt 1 [6, 7, 8] 9` is `[6, 9, 7, 8]`, and
  ||| `insertAt 3 [6, 7, 8] 9` is `[6, 7, 8, 9]`.
  |||
  ||| @idx The index of the value in the resulting list.
  ||| @xs The list to insert the value into.
  ||| @x The value to insert.
  public export
  insertAt : (idx : Nat) -> (x : a) -> (xs : List a) -> {auto 0 prf : idx `LTE` length xs} -> List a
  insertAt Z x xs = x :: xs
  insertAt {prf=LTESucc _} (S n) x (y :: ys) = y :: (insertAt n x ys)

  ||| Delete a value from a list. For example, `deleteAt 1 [3, 4, 5]` is `[3, 5]`.
  |||
  ||| @idx The index of the value to delete.
  ||| @xs The list to delete the value from.
  public export
  deleteAt : (idx : Nat) -> (xs : List a) -> {auto 0 prf : InBounds idx xs} -> List a
  deleteAt {prf=InFirst} Z (_ :: xs) = xs
  deleteAt {prf=InLater _} (S k) (x :: xs) = x :: deleteAt k xs

  ||| Replace an element in a list. For example, `replaceAt 2 6 [1, 2, 3, 4]` is `[1, 2, 6, 4]`.
  |||
  ||| @idx The index of the value to replace.
  ||| @x The value to insert.
  ||| @xs The list in which to replace an element.
  public export
  replaceAt : (idx : Nat) -> a -> (xs : List a) -> {auto 0 prf : InBounds idx xs} -> List a
  replaceAt Z y (_ :: xs) {prf=InFirst} = y :: xs
  replaceAt (S k) y (x :: xs) {prf=InLater _} = x :: replaceAt k y xs
