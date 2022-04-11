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
module Unit.TestLiteral

import Literal

import Utils.Property

test_map : Property
test_map = property $ do
  map (+ 1) (Scalar 2) === Scalar 3
  (map (+ 1) $ the (Literal [0] _) []) === []
  (map (+ 1) $ the (Literal _ _) [[0, 1, 2], [3, 4, 5]]) === [[1, 2, 3], [4, 5, 6]]

test_pure : Property
test_pure = property $ do
  the (Literal [] Nat) (pure 0) === Scalar 0
  the (Literal [0] Nat) (pure 0) === []
  the (Literal [0, 2] Nat) (pure 0) === []
  the (Literal [2, 3] Nat) (pure 0) === [[0, 0, 0], [0, 0, 0]]

test_apply : Property
test_apply = property $ do
  (Scalar (+ 1) <*> Scalar 2) === Scalar 3
  (Scalar (+) <*> Scalar 1 <*> Scalar 2) === Scalar 3
  let f : Literal [0] (() -> ()) = []
      x : Literal [0] () = []
  (f <*> x) === x
  ([Scalar (+ 1), Scalar (+ 1)] <*> [0, 1]) === [1, 2]
  ([Scalar (+), Scalar (+)] <*> [0, 1] <*> [2, 3]) === [2, 4]

test_foldr : Property
test_foldr = property $ do
  let xs : Literal [0] String = []
  foldr (++) "!" xs === "!"

  let xs = Scalar "a"
  foldr (++) "!" xs === "a!"

  let xs = [Scalar "a", Scalar "b", Scalar "c", Scalar "d"]
  foldr String.(++) "!" xs === "abcd!"

  let xs = [[Scalar "a", Scalar "b"], [Scalar "c", Scalar "d"]]
  foldr String.(++) "!" xs === "abcd!"

test_all : Property
test_all = property $ do
  all True === True
  all False === False
  all (the (Literal [0] Bool) []) === True
  all [True, True] === True
  all [True, False] === False
  all [False, False] === False

test_show : Property
test_show = property $ do
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

test_cast : Property
test_cast = property $ do
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

export
root : Group
root = MkGroup "Literal" $ [
      ("map", test_map)
    , ("pure", test_pure)
    , ("<*>", test_apply)
    , ("foldr", test_foldr)
    , ("all", test_all)
    , ("show", test_show)
  ]
