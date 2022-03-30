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

test_pure : IO ()
test_pure = do
  assert "Literal pure" $ the (Literal [2, 3] Nat) (pure 0) == [[0, 0, 0], [0, 0, 0]]

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

export
test : IO ()
test = do
  test_pure
  test_foldr
