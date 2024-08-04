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

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

partial
vmap : Property
vmap = fixedProperty $ do
  let xs = tensor {dtype=S32} [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
      x = tensor {dtype=S32} [[1, 0], [-1, 2]]

  -- unary
  (do vmap diag !xs) ===# tensor [[0, 3], [4, 7]]
  (do vmap (\_ => x) !xs) ===# tensor [[[1, 0], [-1, 2]], [[1, 0], [-1, 2]]]
  (do vmap (\_ => diag !x) !xs) ===# tensor [[1, 2], [1, 2]]
  (do vmap (\_ => do diag !x) !xs) ===# tensor [[1, 2], [1, 2]]

  (do vmap (expand 0) !xs) ===# (do expand 1 !xs)
  (do vmap (\_ => expand 0 !x) !xs) ===# tensor [[[[1, 0], [-1, 2]]], [[[1, 0], [-1, 2]]]]

  -- binary
  (do vmap (\x => concat 0 x x) !xs) ===# (do concat 1 !xs !xs)

  (do vmap (\x => concat 0 !(tensor [[8, 9]]) x) !xs) ===# tensor {dtype = S32} [
    [[8, 9], [0, 1], [2, 3]],
    [[8, 9], [4, 5], [6, 7]]
  ]
{-
  (do vmap (\x => concat 0 x !(tensor [[8, 9]])) !xs) ===# tensor {dtype = S32} [
    [[0, 1], [2, 3], [8, 9]],
    [[4, 5], [6, 7], [8, 9]]
  ]
  (do vmap (\_ => concat 0 !(tensor [0]) !(tensor [1])) !xs) ===# tensor {dtype = S32} [[0, 1], [0, 1]]

  (do vmap (\x => pure x + pure x) !xs) ===# xs + xs
  (do vmap (\x => diag !(tensor [[1, -1], [2, -3]]) + diag x) !xs) ===# tensor [[1, 0], [5, 4]]
  (do vmap (\x => vmap (\y => concat 0 !(expand 0 y) x) x) !xs) ===# tensor [
    [[[0, 1], [0, 1], [2, 3]], [[2, 3], [0, 1], [2, 3]]],
    [[[4, 5], [4, 5], [6, 7]], [[6, 7], [4, 5], [6, 7]]]
  ]
-}



{-
      y = fromLiteral {dtype=S32} [[4, -2], [5, 1]]
  vmap (\x => x - y) xs ===# fromLiteral [[[-4, 3], [-3, 2]], [[0, 7], [1, 2]]]
  vmap (y -) xs ===# fromLiteral [[[4, -3], [3, -2]], [[0, -7], [-1, -2]]]
  vmap (+ y) xs ===# fromLiteral [[[4, -1], [7, 4]], [[8, 3], [11, 4]]]
  vmap (y +) xs ===# fromLiteral [[[4, -1], [7, 4]], [[8, 3], [11, 4]]]
  vmap (const y) xs ===# broadcast y

  vmap (\x => concat 0 y x) xs ===# fromLiteral [
      [[4, -2], [5, 1], [0, 1], [2, 3]], [[4, -2], [5, 1], [4, 5], [6, 3]]
    ]
  vmap (\x => concat 1 x y) xs ===# fromLiteral [
      [[0, 1, 4, -2], [2, 3, 5, 1]], [[4, 5, 4, -2], [6, 3, 5, 1]]
    ]

  vmap (\x => reduce @{Sum} [0] x) xs ===# fromLiteral [[2, 4], [10, 8]]

  let preds = fromLiteral [True, False]
  vmap (\x => cond x id 1 id 0) preds ===# fromLiteral [1, 0]
  vmap (\x => cond (fromLiteral True) id x id (fill {shape=[2, 2]} 0)) xs ===# xs
  vmap (\x => cond (fromLiteral True) (const x) (fill {shape=[]} {dtype=U32} 1) id (fill 0)) xs
    ===# xs

  -- [[2, 3], [0, 1]] + [[0, 3], [4, -2]]
  -- [[6, 3], [4, 5]] + [[4, 3]], [4, -2]]
  vmap (\x => reverse [0] x + concat 0 (expand 0 (diag x)) (slice [0.to 1] y)) xs ===#
    fromLiteral [[[2, 6], [4, -1]], [[10, 6], [8, 3]]]

  let a = fromLiteral [[[1.0, 0.0], [-3.0, 2.2]], [[-2.0, 0.0], [-2.5, 1.5]]]
      x = fromLiteral [[1.1, -1.2], [2.0, 2.2]]
      b = fromLiteral [[1.1, -5.94], [-4.0, -1.7]]
  vmap (|\) a b ===# x
-}

partial
reduce : Property
reduce = fixedProperty $ do
  let x = tensor {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [1] !x) ===# tensor [6, -6]

  let x = tensor {dtype=S32} [[1, 2, 3], [-2, -3, -4]]
  (do reduce @{Sum} [0, 1] !x) ===# tensor (-3)

  let x = tensor {dtype=S32} [[[1], [2], [3]], [[-2], [-3], [-4]]]
  (do reduce @{Sum} [0, 1] !x) ===# tensor [-3]

  let x = tensor {dtype=S32} [[[1, 2, 3]], [[-2, -3, -4]]]
  (do reduce @{Sum} [0, 2] !x) ===# tensor [-3]

  let x = tensor {dtype=S32} [[[1, 2, 3], [-2, -3, -4]]]
  (do reduce @{Sum} [1, 2] !x) ===# tensor [-3]

  let x = tensor {dtype=S32} [[[1, 2, 3], [4, 5, 6]], [[-2, -3, -4], [-6, -7, -8]]]
  (do reduce @{Sum} [0, 2] !x) ===# tensor [-3, -6]

  let x = tensor {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [0] !x) ===# tensor [0, 0, 0]

  let x = tensor {dtype=PRED} [[True, False, True], [True, False, False]]
  (do reduce @{All} [1] !x) ===# tensor [False, False]

partial
sort : Property
sort = withTests 20 . property $ do
  d <- forAll dims
  dd <- forAll dims
  ddd <- forAll dims

  x <- forAll (literal [S d] int32s)
  let x = tensor {dtype=S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd] int32s)
  let x = tensor {dtype=S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  let sorted = (do sort (<) 1 !x)
      init = (do slice [all, 0.to dd] !sorted)
      tail = (do slice [all, 1.to (S dd)] !sorted)
  diff (unsafeEval init) (\x, y => all [| x <= y |]) (unsafeEval tail)

  x <- forAll (literal [S d, S dd, S ddd] int32s)
  let x = tensor {dtype=S32} x

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
  lteSucc = lteSuccRight (reflexive {ty=Nat})

  %hint
  reflex : {n : _} -> LTE n n
  reflex = reflexive {ty=Nat}

partial
sortWithEmptyAxis : Property
sortWithEmptyAxis = fixedProperty $ do
  let x = tensor {shape=[0, 2, 3]} {dtype=S32} []
  (do sort (<) 0 !x) ===# x

  let x = tensor {shape=[0, 2, 3]} {dtype=S32} []
  (do sort (<) 1 !x) ===# x

  let x = tensor {shape=[2, 0, 3]} {dtype=S32} [[], []]
  (do sort (<) 0 !x) ===# x

  let x = tensor {shape=[2, 0, 3]} {dtype=S32} [[], []]
  (do sort (<) 1 !x) ===# x

partial
sortWithRepeatedElements : Property
sortWithRepeatedElements = fixedProperty $ do
  let x = tensor {dtype=S32} [1, 3, 4, 3, 2]
  (do sort (<) 0 !x) ===# tensor [1, 2, 3, 3, 4]

  let x = tensor {dtype=S32} [[1, 4, 4], [3, 2, 5]]
  (do sort (<) 0 !x) ===# tensor [[1, 2, 4], [3, 4, 5]]
  (do sort (<) 1 !x) ===# tensor [[1, 4, 4], [2, 3, 5]]

partial
condResultTrivialUsage : Property
condResultTrivialUsage = fixedProperty $ do
  let x = tensor {dtype=S32} 0
  (do cond !(tensor True) (\x => pure x + 1) !x (\x => pure x - 1) !x) ===# 1

  let x = tensor {dtype=S32} 0
  (do cond !(tensor False) (\x => pure x + 1) !x (\x => pure x - 1) !x) ===# -1

  let x = tensor {dtype=S32} [2, 3]
      y = tensor [[6, 7], [8, 9]]
  (do cond !(tensor True) (\x => tensor 5 * pure x) !x diag !y) ===# tensor [10, 15]

  let x = tensor {dtype=S32} [2, 3]
      y = tensor [[6, 7], [8, 9]]
  (do cond !(tensor False) (\x => tensor 5 * pure x) !x diag !y) ===# tensor [6, 9]

partial
condResultWithReusedArgs : Property
condResultWithReusedArgs = fixedProperty $ do
  let x = tensor {dtype=S32} 1
      y = tensor {dtype=S32} 3

      f : (Graph a -> Graph a -> Graph a) -> a -> Graph a
      f g x = g (pure x) (pure x)

  (do cond !(tensor True) (f (+)) !x (f (*)) !y) ===# 2
  (do cond !(tensor False) (f (+)) !x (f (*)) !y) ===# 9

export partial
all : List (PropertyName, Property)
all = [
      ("vmap", vmap){-
    , ("reduce", reduce)
    , ("sort", sort)
    , ("sort with empty axis", sortWithEmptyAxis)
    , ("sort with repeated elements", sortWithRepeatedElements)
    , ("cond for trivial usage", condResultTrivialUsage)
    , ("cond for re-used arguments", condResultWithReusedArgs)-}
  ]
