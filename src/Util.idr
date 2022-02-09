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

import Data.Vect
import Data.Nat

namespace Vect
  ||| All numbers from `0` to `n - 1` inclusive, in increasing order.
  |||
  ||| @n The (exclusive) limit of the range.
  export
  range : (n : Nat) -> Vect n Nat
  range Z = []
  range (S n) = snoc (range n) n

namespace List
  ||| All numbers from `0` to `n - 1` inclusive, in increasing order.
  |||
  ||| @n The (exclusive) limit of the range.
  export
  range : (n : Nat) -> List Nat
  range n = toList (Vect.range n)

||| Insert a value in a list.
|||
||| @idx The index of the value in the resulting list.
||| @xs The list to insert the value into.
||| @x The value to insert.
public export
insertAt : (idx : Nat) -> (xs : List a) -> (x : a) -> {auto prf : idx `LTE` length xs} -> List a
insertAt Z xs x = x :: xs
insertAt {prf=LTESucc _} (S n) (y :: ys) x = y :: (insertAt n ys x)

||| Delete a value from a list.
|||
||| @idx The index of the value to delete.
||| @xs The list to delete the value from.
public export
deleteAt : (idx : Nat) -> (xs : List a) -> {auto prf : idx `LT` length xs} -> List a
deleteAt Z (_ :: xs) = xs
deleteAt {prf=LTESucc _} (S n) (x :: xs) = x :: (deleteAt n xs)
