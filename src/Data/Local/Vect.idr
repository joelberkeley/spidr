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
||| This module extends the standard library's Data.Vect
module Data.Local.Vect

import public Data.Vect

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
