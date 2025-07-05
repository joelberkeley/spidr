{--
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
module Unit.TestLiteral

import Literal

import Utils.Comparison
import Utils.Cases

map : Property
map = fixedProperty $ do
  map (+ 1) (Scalar 2) === Scalar 3
  (map (+ 1) $ the (Literal [0] Nat) []) === []
  (map (+ 1) $ the (Literal _ _) [[0, 1, 2], [3, 4, 5]]) === [[1, 2, 3], [4, 5, 6]]

pure : Property
pure = fixedProperty $ do
  the (Literal [] Nat) (pure 0) === Scalar 0
  the (Literal [0] Nat) (pure 0) === []
  the (Literal [0, 2] Nat) (pure 0) === []
  the (Literal [2, 3] Nat) (pure 0) === [[0, 0, 0], [0, 0, 0]]

(<*>) : Property
(<*>) = fixedProperty $ do
  (Scalar (+ 1) <*> Scalar 2) === Scalar 3
  (Scalar (+) <*> Scalar 1 <*> Scalar 2) === Scalar 3
  let f : Literal [0] (() -> ()) = []
      x : Literal [0] () = []
  (f <*> x) === x
  ([Scalar (+ 1), Scalar (+ 1)] <*> [0, 1]) === [1, 2]
  ([Scalar (+), Scalar (+)] <*> [0, 1] <*> [2, 3]) === [2, 4]

foldr : Property
foldr = fixedProperty $ do
  let xs : Literal [0] String = []
  foldr (++) "!" xs === "!"

  let xs = Scalar "a"
  foldr (++) "!" xs === "a!"

  let xs = [Scalar "a", Scalar "b", Scalar "c", Scalar "d"]
  foldr String.(++) "!" xs === "abcd!"

  let xs = [[Scalar "a", Scalar "b"], [Scalar "c", Scalar "d"]]
  foldr String.(++) "!" xs === "abcd!"

all : Property
all = fixedProperty $ do
  all True === True
  all False === False
  all (the (Literal [0] Bool) []) === True
  all [True, True] === True
  all [True, False] === False
  all [False, False] === False

show : Property
show = fixedProperty $ do
  show (Scalar $ the Int 1) === "1"
  show (Scalar $ the Double 1.2) === "1.2"
  show Literal.True === "True"
  show (the (Literal _ _) [0, 1, 2]) === "[0, 1, 2]"
  show (the (Literal _ _) [[0, 1, 2], [3, 4, 5]]) === "[[0, 1, 2],\n [3, 4, 5]]"
  let expected = "[[0.1, 1.1, 2.1],\n [-3.1, 4.1, 5.1]]"
  show (the (Literal _ _) [[0.1, 1.1, 2.1], [-3.1, 4.1, 5.1]]) === expected
  show (the (Literal _ _) [[True, False]]) === "[[True, False]]"

  show (the (Literal [0] Nat) []) === "[]"
  show (the (Literal [0, 0] Nat) []) === "[]"
  show (the (Literal [0, 1] Nat) []) === "[]"
  show (the (Literal [1, 0] Nat) [[]]) === "[[]]"
  show (the (Literal [1, 0, 1] Nat) [[]]) === "[[]]"
  show (the (Literal [2, 0] Nat) [[], []]) === "[[],\n []]"
  
  let xs : Literal [3, 2, 2] Nat = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
  show xs === "[[[0, 1],\n  [2, 3]],\n [[4, 5],\n  [6, 7]],\n [[8, 9],\n  [10, 11]]]"

cast : Property
cast = fixedProperty $ do
  let lit : Literal [] Nat = Scalar 1
      arr : Array [] Nat = 1
  cast @{toArray} lit === arr
  lit === cast arr

  let lit : Literal [0] Nat = []
      arr : Array [0] Nat = []
  cast @{toArray} lit === arr
  lit === cast arr

  let lit : Literal [2, 3] Nat = [[0, 1, 2], [3, 4, 5]]
      arr : Array [2, 3] Nat = [[0, 1, 2], [3, 4, 5]]
  cast @{toArray} lit === arr
  lit === cast arr

scalarZeroNotSucc : All IsSucc (Scalar 0) -> Void
scalarZeroNotSucc (Scalar _) impossible

scalarZeroVectNotSucc0 : All IsSucc [Scalar 0, Scalar 1] -> Void
scalarZeroVectNotSucc0 (x :: _) = scalarZeroNotSucc x

scalarZeroVectNotSucc1 : All IsSucc [Scalar 1, Scalar 0] -> Void
scalarZeroVectNotSucc1 (_ :: x :: _) = scalarZeroNotSucc x

export
group : Group
group = MkGroup "Literal" $ [
      ("map", map)
    , ("pure", pure)
    , ("(<*>)", (<*>))
    , ("foldr", foldr)
    , ("all", all)
    , ("show", show)
  ]
