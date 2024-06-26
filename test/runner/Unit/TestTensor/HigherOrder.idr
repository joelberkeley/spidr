{--
Copyright 2023 Joel Berkeley

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
module Unit.TestTensor.HigherOrder

import System

import Device
import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

partial
mapResult : Device => Property
mapResult = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = tensor {dtype = F64} x
  map id x ==~ unsafeEval (do map pure !x')
  map (1.0 /) x ==~ unsafeEval (do map (\x => 1.0 / pure x) !x')

  x <- forAll (literal shape int32s)
  let x' = tensor {dtype = S32} x
  map (+ 1) x === unsafeEval (do map (\x => pure x + 1) !x')

partial
mapNonTrivial : Device => Property
mapNonTrivial = fixedProperty $ do
  (do map {a = S32} (\x => pure x + pure x) !1) ===# 2
  (do map {a = S32} (\_ => 2) !1) ===# 2
  (do map {a = S32} (map (\x => pure x + 1)) !1) ===# 2

partial
map2Result : Device => Property
map2Result = fixedProperty $ do
  shape <- forAll shapes

  let int32s = literal shape int32s
  [x, y] <- forAll (np [int32s, int32s])
  let x' = tensor {dtype = S32} x
      y' = tensor {dtype = S32} y
  [| x + y |] === unsafeEval (do map2 (\x, y => pure x + pure y) !x' !y')

  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = tensor {dtype = F64} x
      y' = tensor {dtype = F64} y
  [| x + y |] ==~ unsafeEval (do map2 (\x, y => pure x + pure y) !x' !y')

partial
map2ResultWithReusedFnArgs : Device => Property
map2ResultWithReusedFnArgs = fixedProperty $ do
  let x : Graph (Tensor [] S32) = 6
  (do map2 (\x, y => pure x + pure x + pure y + pure y) !1 !2) ===# x

partial
reduce : Device => Property
reduce = fixedProperty $ do
  let x = tensor {dtype = S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [1] !x) ===# tensor [6, -6]

  let x = tensor {dtype = S32} [[1, 2, 3], [-2, -3, -4]]
  (do reduce @{Sum} [0, 1] !x) ===# tensor (-3)

  let x = tensor {dtype = S32} [[[1], [2], [3]], [[-2], [-3], [-4]]]
  (do reduce @{Sum} [0, 1] !x) ===# tensor [-3]

  let x = tensor {dtype = S32} [[[1, 2, 3]], [[-2, -3, -4]]]
  (do reduce @{Sum} [0, 2] !x) ===# tensor [-3]

  let x = tensor {dtype = S32} [[[1, 2, 3], [-2, -3, -4]]]
  (do reduce @{Sum} [1, 2] !x) ===# tensor [-3]

  let x = tensor {dtype = S32} [[[1, 2, 3], [4, 5, 6]], [[-2, -3, -4], [-6, -7, -8]]]
  (do reduce @{Sum} [0, 2] !x) ===# tensor [-3, -6]

  let x = tensor {dtype = S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [0] !x) ===# tensor [0, 0, 0]

  let x = tensor {dtype = PRED} [[True, False, True], [True, False, False]]
  (do reduce @{All} [1] !x) ===# tensor [False, False]

partial
sort : Device => Property
sort = withTests 20 . property $ do
  d <- forAll dims
  dd <- forAll dims
  ddd <- forAll dims

  x <- forAll (literal [S d] int32s)
  let x = tensor {dtype = S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd] int32s)
  let x = tensor {dtype = S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = (do sort (<) 1 !x)
      init = (do slice [all, 0.to dd] !sorted)
      tail = (do slice [all, 1.to (S dd)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd, S ddd] int32s)
  let x = tensor {dtype = S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = (do sort (<) 1 !x)
      init = (do slice [all, 0.to dd] !sorted)
      tail = (do slice [all, 1.to (S dd)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = (do sort (<) 2 !x)
      init = (do slice [all, all, 0.to ddd] !sorted)
      tail = (do slice [all, all, 1.to (S ddd)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  where
  %hint
  lteSucc : {n : _} -> LTE n (S n)
  lteSucc = lteSuccRight (reflexive {ty = Nat})

  %hint
  reflex : {n : _} -> LTE n n
  reflex = reflexive {ty = Nat}

partial
sortWithEmptyAxis : Device => Property
sortWithEmptyAxis = fixedProperty $ do
  let x = tensor {shape = [0, 2, 3]} {dtype = S32} []
  (do sort (<) 0 !x) ===# x

  let x = tensor {shape = [0, 2, 3]} {dtype = S32} []
  (do sort (<) 1 !x) ===# x

  let x = tensor {shape = [2, 0, 3]} {dtype = S32} [[], []]
  (do sort (<) 0 !x) ===# x

  let x = tensor {shape = [2, 0, 3]} {dtype = S32} [[], []]
  (do sort (<) 1 !x) ===# x

partial
sortWithRepeatedElements : Device => Property
sortWithRepeatedElements = fixedProperty $ do
  let x = tensor {dtype = S32} [1, 3, 4, 3, 2]
  (do sort (<) 0 !x) ===# tensor [1, 2, 3, 3, 4]

  let x = tensor {dtype = S32} [[1, 4, 4], [3, 2, 5]]
  (do sort (<) 0 !x) ===# tensor [[1, 2, 4], [3, 4, 5]]
  (do sort (<) 1 !x) ===# tensor [[1, 4, 4], [2, 3, 5]]

partial
condResultTrivialUsage : Device => Property
condResultTrivialUsage = fixedProperty $ do
  let x = tensor {dtype = S32} 0
  (do cond !(tensor True) (\x => pure x + 1) !x (\x => pure x - 1) !x) ===# 1

  let x = tensor {dtype = S32} 0
  (do cond !(tensor False) (\x => pure x + 1) !x (\x => pure x - 1) !x) ===# -1

  let x = tensor {dtype = S32} [2, 3]
      y = tensor [[6, 7], [8, 9]]
  (do cond !(tensor True) (\x => tensor 5 * pure x) !x diag !y) ===# tensor [10, 15]

  let x = tensor {dtype = S32} [2, 3]
      y = tensor [[6, 7], [8, 9]]
  (do cond !(tensor False) (\x => tensor 5 * pure x) !x diag !y) ===# tensor [6, 9]

partial
condResultWithReusedArgs : Device => Property
condResultWithReusedArgs = fixedProperty $ do
  let x = tensor {dtype = S32} 1
      y = tensor {dtype = S32} 3

      f : (Graph a -> Graph a -> Graph a) -> a -> Graph a
      f g x = g (pure x) (pure x)

  (do cond !(tensor True) (f (+)) !x (f (*)) !y) ===# 2
  (do cond !(tensor False) (f (+)) !x (f (*)) !y) ===# 9

export partial
all : Device => List (PropertyName, Property)
all = [
      ("map", mapResult)
    , ("map with non-trivial function", mapNonTrivial)
    , ("map2", map2Result)
    , ("map2 with re-used function arguments", map2ResultWithReusedFnArgs)
    , ("reduce", reduce)
    , ("sort", sort)
    , ("sort with empty axis", sortWithEmptyAxis)
    , ("sort with repeated elements", sortWithRepeatedElements)
    , ("cond for trivial usage", condResultTrivialUsage)
    , ("cond for re-used arguments", condResultWithReusedArgs)
  ]
