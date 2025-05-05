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
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

mapResult : Device => Property
mapResult = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = tensor {dtype = F64} x
  map id x ==~ unsafeEval (map pure x')
  map (1.0 /) x ==~ Tag.unsafeEval (map (pure . (1.0 /)) x')

  x <- forAll (literal shape int32s)
  let x' = tensor {dtype = S32} x
  map (+ 1) x === Tag.unsafeEval (map (pure . (+ 1)) x')

mapNonTrivial : Device => Property
mapNonTrivial = fixedProperty $ do
  let res : Tag (Tensor [] S32) = pure 2
  map {a = S32} (\x => pure $ x + x) 1 ===# res
  map {a = S32} (\_ => pure 2) 1 ===# res
  map {a = S32} (map (\x => pure $ x + 1)) 1 ===# res

  let x : Tag $ Tensor [] S32 = do
      tag =<< Tensor.map (\y => do
          Prelude.map (y +) $ tag =<< Tensor.map (
              \u => do v <- tag (tensor 3); pure $ u * v
            ) y
        ) (tensor 7)
  x ===# pure 28

map2Result : Device => Property
map2Result = fixedProperty $ do
  shape <- forAll shapes

  let int32s = literal shape int32s
  [x, y] <- forAll (np [int32s, int32s])
  let x' = tensor {dtype = S32} x
      y' = tensor {dtype = S32} y
  [| x + y |] === unsafeEval (map2 (pure .: Tensor.(+)) x' y')

  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = tensor {dtype = F64} x
      y' = tensor {dtype = F64} y
  [| x + y |] ==~ unsafeEval (map2 (pure .: Tensor.(+)) x' y')

map2ResultWithReusedFnArgs : Device => Property
map2ResultWithReusedFnArgs = fixedProperty $ do
  let x : Tensor [] S32 = 6
  map2 (\x, y => pure $ x + x + y + y) 1 2 ===# pure x

reduce : Device => Property
reduce = fixedProperty $ do
  let x = tensor {dtype = S32} [[1, 2, 3], [-1, -2, -3]]
  reduce @{Sum} [1] x ===# pure (tensor [6, -6])

  let x = tensor {dtype = S32} [[1, 2, 3], [-2, -3, -4]]
  reduce @{Sum} [0, 1] x ===# pure (tensor (-3))

  let x = tensor {dtype = S32} [[[1], [2], [3]], [[-2], [-3], [-4]]]
  reduce @{Sum} [0, 1] x ===# pure (tensor [-3])

  let x = tensor {dtype = S32} [[[1, 2, 3]], [[-2, -3, -4]]]
  reduce @{Sum} [0, 2] x ===# pure (tensor [-3])

  let x = tensor {dtype = S32} [[[1, 2, 3], [-2, -3, -4]]]
  reduce @{Sum} [1, 2] x ===# pure (tensor [-3])

  let x = tensor {dtype = S32} [[[1, 2, 3], [4, 5, 6]], [[-2, -3, -4], [-6, -7, -8]]]
  reduce @{Sum} [0, 2] x ===# pure (tensor [-3, -6])

  let x = tensor {dtype = S32} [[1, 2, 3], [-1, -2, -3]]
  reduce @{Sum} [0] x ===# pure (tensor [0, 0, 0])

  let x = tensor {dtype = PRED} [[True, False, True], [True, False, False]]
  reduce @{All} [1] x ===# pure (tensor [False, False])

sort : Device => Property
sort = withTests 20 . property $ do
  d <- forAll dims
  dd <- forAll dims
  ddd <- forAll dims

  x <- forAll (literal [S d] int32s)
  let x = tensor {dtype = S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] <$> sorted
      tail = slice [1.to (S d)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd] int32s)
  let x = tensor {dtype = S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] <$> sorted
      tail = slice [1.to (S d)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = sort (<) 1 x
      init = slice [all, 0.to dd] <$> sorted
      tail = slice [all, 1.to (S dd)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd, S ddd] int32s)
  let x = tensor {dtype = S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] <$> sorted
      tail = slice [1.to (S d)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = sort (<) 1 x
      init = slice [all, 0.to dd] <$> sorted
      tail = slice [all, 1.to (S dd)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = sort (<) 2 x
      init = slice [all, all, 0.to ddd] <$> sorted
      tail = slice [all, all, 1.to (S ddd)] <$> sorted
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  where
  %hint
  lteSucc : {n : _} -> LTE n (S n)
  lteSucc = lteSuccRight (reflexive {ty = Nat})

  %hint
  reflex : {n : _} -> LTE n n
  reflex = reflexive {ty = Nat}

sortWithEmptyAxis : Device => Property
sortWithEmptyAxis = fixedProperty $ do
  let x = tensor {shape = [0, 2, 3]} {dtype = S32} []
  sort (<) 0 x ===# pure x

  let x = tensor {shape = [0, 2, 3]} {dtype = S32} []
  sort (<) 1 x ===# pure x

  let x = tensor {shape = [2, 0, 3]} {dtype = S32} [[], []]
  sort (<) 0 x ===# pure x

  let x = tensor {shape = [2, 0, 3]} {dtype = S32} [[], []]
  sort (<) 1 x ===# pure x

sortWithRepeatedElements : Device => Property
sortWithRepeatedElements = fixedProperty $ do
  let x = tensor {dtype = S32} [1, 3, 4, 3, 2]
  sort (<) 0 x ===# pure (tensor [1, 2, 3, 3, 4])

  let x = tensor {dtype = S32} [[1, 4, 4], [3, 2, 5]]
  sort (<) 0 x ===# pure (tensor [[1, 2, 4], [3, 4, 5]])
  sort (<) 1 x ===# pure (tensor [[1, 4, 4], [2, 3, 5]])

condResultTrivialUsage : Device => Property
condResultTrivialUsage = fixedProperty $ do
  let x = tensor {dtype = S32} 0
  cond (tensor True) (\x => pure $ x + 1) x (\x => pure $ x - 1) x ===# pure (the (Tensor [] S32) 1)

  let x = tensor {dtype = S32} 0
  cond (tensor False) (\x => pure $ x + 1) x (\x => pure $ x - 1) x ===#
    pure (the (Tensor [] S32) $ -1)

  let x = tensor {dtype = S32} [2, 3]
      y = tensor {dtype = S32} [[6, 7], [8, 9]]
  cond (tensor True) (\x => pure $ tensor 5 * x) x (pure . diag) y ===# pure (tensor [10, 15])

  let x = tensor {dtype = S32} [2, 3]
      y = tensor {dtype = S32} [[6, 7], [8, 9]]
  cond (tensor False) (\x => pure $ tensor 5 * x) x (pure . diag) y ===# pure (tensor [6, 9])

condResultWithReusedArgs : Device => Property
condResultWithReusedArgs = fixedProperty $ do
  let x = tensor {dtype = S32} 1
      y = tensor {dtype = S32} 3

      f : (a -> a -> a) -> a -> Tag a
      f g x = pure $ g x x

  cond (tensor True) (f (+)) x (f (*)) y ===# pure 2
  cond (tensor False) (f (+)) x (f (*)) y ===# pure 9

  let f : Taggable a => (a -> a -> a) -> a -> Tag a
      f g x = tag x <&> \x => g x x

  cond (tensor True) (f (+)) x (f (*)) y ===# pure 2
  cond (tensor False) (f (+)) x (f (*)) y ===# pure 9

while : Device => Property
while = fixedProperty $ do
  let initial = tensor {dtype = S32} 10
      condition = \x => pure $ x * x > 4
      body = \x => pure $ x - 1

  while condition body initial ===# pure 2

  let initial = tensor {dtype = S32} 4
      condition = \x => pure $ x < 4
      body = \x => pure $ 2 * x

  while condition body initial ===# pure 4

  let initial = tensor {dtype = S32} 4
      condition = \x => pure $ x <= 4
      body = \x => pure $ 2 * x

  while condition body initial ===# pure 8

  let initial = tensor [1.0, 2.0]
      condition = \x => pure $ slice [at 0] x < 30.0
      body = \x : Tensor [2] F64 => pure $ 2.0 * x

  while condition body initial ===# pure (tensor [32.0, 64.0])

export
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
    , ("while", while)
  ]
