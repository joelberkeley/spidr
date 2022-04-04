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

import Utils

import Literal

test_map : IO ()
test_map = do
  assert "Literal map scalar" $ map (+ 1) (Scalar 2) == Scalar 3
  assert "Literal map empty" $ (map (+ 1) $ the (Literal [0] _) []) == []
  assert "Literal map array" $
    (map (+ 1) $ the (Literal _ _) [[0, 1, 2], [3, 4, 5]]) == [[1, 2, 3], [4, 5, 6]]

test_pure : IO ()
test_pure = do
  assert "Literal pure scalar" $ the (Literal [] Nat) (pure 0) == Scalar 0
  assert "Literal pure empty vector" $ the (Literal [0] Nat) (pure 0) == []
  assert "Literal pure empty array" $ the (Literal [0, 2] Nat) (pure 0) == []
  assert "Literal pure array" $ the (Literal [2, 3] Nat) (pure 0) == [[0, 0, 0], [0, 0, 0]]

test_apply : IO ()
test_apply = do
  assert "Literal (<*>) scalar unary" $ (Scalar (+ 1) <*> Scalar 2) == Scalar 3
  assert "Literal (<*>) scalar binary" $ (Scalar (+) <*> Scalar 1 <*> Scalar 2) == Scalar 3
  assert "Literal (<*>) empty" $
    let f : Literal [0] (() -> ()) = []
        x : Literal [0] () = []
     in (f <*> x) == x
  assert "Literal (<*>) array unary" $ ([Scalar (+ 1), Scalar (+ 1)] <*> [0, 1]) == [1, 2]
  assert "Literal (<*>) array binary" $ ([Scalar (+), Scalar (+)] <*> [0, 1] <*> [2, 3]) == [2, 4]

test_foldr : IO ()
test_foldr = do
  let xs : Literal [0] String = []
  assert "Literal foldr empty" $ foldr (++) "!" xs == "!"

  let xs = Scalar "a"
  assert "Literal foldr scalar" $ foldr (++) "!" xs == "a!"

  let xs = [Scalar "a", Scalar "b", Scalar "c", Scalar "d"]
  assert "Literal foldr vector" $ foldr String.(++) "!" xs == "abcd!"

  let xs = [[Scalar "a", Scalar "b"], [Scalar "c", Scalar "d"]]
  assert "Literal foldr matrix" $ foldr String.(++) "!" xs == "abcd!"

test_all : IO ()
test_all = do
  assert "Literal all scalar true" $ all True == True
  assert "Literal all scalar false" $ all False == False
  assert "Literal all empty" $ all (the (Literal [0] Bool) []) == True
  assert "Literal all array all true" $ all [True, True] == True
  assert "Literal all array some false" $ all [True, False] == False
  assert "Literal all array all false" $ all [False, False] == False

test_show : IO ()
test_show = do
  assert "Literal show scalar Int" $ show (Scalar $ the Int 1) == "1"
  assert "Literal show scalar Double" $ show (Scalar $ the Double 1.2) == "1.2"
  assert "Literal show scalar Bool" $ show Literal.True == "True"
  assert "Literal show vector Int" $ show (the (Literal _ _) [0, 1, 2]) == "[0, 1, 2]"
  assert "Literal show array Int" $
    show (the (Literal _ _) [[0, 1, 2], [3, 4, 5]]) == "[[0, 1, 2],\n [3, 4, 5]]"
  assert "Literal show array Double" $
    let expected = "[[0.1, 1.1, 2.1],\n [-3.1, 4.1, 5.1]]"
     in show (the (Literal _ _) [[0.1, 1.1, 2.1], [-3.1, 4.1, 5.1]]) == expected
  assert "Literal show array Bool" $ show (the (Literal _ _) [[True, False]]) == "[[True, False]]"
  
  assert "Literal show shape [0]" $ show (the (Literal [0] Nat) []) == "[]"
  assert "Literal show shape [0, 0]" $ show (the (Literal [0, 0] Nat) []) == "[]"
  assert "Literal show shape [0, 1]" $ show (the (Literal [0, 1] Nat) []) == "[]"
  assert "Literal show shape [1, 0]" $ show (the (Literal [1, 0] Nat) [[]]) == "[[]]"
  assert "Literal show shape [1, 0, 1]" $ show (the (Literal [1, 0, 1] Nat) [[]]) == "[[]]"
  assert "Literal show shape [2, 0]" $ show (the (Literal [2, 0] Nat) [[], []]) == "[[],\n []]"
  assert "Literal show shape [3, 2, 2]" $
    let xs : Literal [3, 2, 2] Nat = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
     in show xs == "[[[0, 1],\n  [2, 3]],\n [[4, 5],\n  [6, 7]],\n [[8, 9],\n  [10, 11]]]"

test_cast : IO ()
test_cast = do
  let lit : Literal [] Nat = Scalar 1
      arr : Array [] Nat = 1
  assert "Literal cast scalar to Array" $ cast @{toArray} lit == arr
  assert "Literal cast scalar from Array" $ lit == cast arr

  let lit : Literal [0] Nat = []
      arr : Array [0] Nat = []
  assert "Literal cast empty to Array" $ cast @{toArray} lit == arr
  assert "Literal cast empty from Array" $ lit == cast arr

  let lit : Literal [2, 3] Nat = [[0, 1, 2], [3, 4, 5]]
      arr : Array [2, 3] Nat = [[0, 1, 2], [3, 4, 5]]
  assert "Literal cast array to Array" $ cast @{toArray} lit == arr
  assert "Literal cast array from Array" $ lit == cast arr

export
test : IO ()
test = do
  test_map
  test_pure
  test_apply
  test_foldr
  test_all
  test_show
