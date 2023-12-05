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
module Unit.TestTensor.Slice

import System

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

namespace MultiSlice
  indexFirstDim :
    (n, idx : Nat) ->
    (shape : Shape) ->
    LT idx n ->
    MultiSlice.slice {shape=n :: shape} [at idx] === shape
  indexFirstDim n idx shape x = Refl

  sliceFirstDim :
    (n, from, size : Nat) ->
    (shape : Shape) ->
    LTE (from + size) n ->
    MultiSlice.slice {shape=n :: shape} [from.to {size} (from + size)] === (size :: shape)
  sliceFirstDim n from size shape x = Refl

  export
  slice : Property
  slice = fixedProperty $ do
    slice {shape=[3, 4]} [0.to 3, 0.to 0] === [3, 0]
    slice {shape=[3, 4]} [0.to 3, 0.to 1] === [3, 1]
    slice {shape=[3, 4]} [0.to 3, 0.to 4] === [3, 4]

    slice {shape=[3, 4]} [at 1, 0.to 3] === [3]
    slice {shape=[3, 4]} [0.to 2, at 2] === [2]
    slice {shape=[3, 4]} [at 1, at 2] === Prelude.Nil

partial
sliceStaticIndex : Property
sliceStaticIndex = fixedProperty $ do
  let x = tensor {dtype=S32} [3, 4, 5]
  (do slice [at 0] !x) ===# tensor 3
  (do slice [at 1] !x) ===# tensor 4
  (do slice [at 2] !x) ===# tensor 5

  let idx : Nat
      idx = 2

  (do slice [at idx] !x) ===# tensor 5

partial
sliceStaticSlice : Property
sliceStaticSlice = fixedProperty $ do
  let x = tensor {dtype=S32} [3, 4, 5]
  (do slice [0.to 0] !x) ===# tensor []
  (do slice [0.to 1] !x) ===# tensor [3]
  (do slice [0.to 2] !x) ===# tensor [3, 4]
  (do slice [0.to 3] !x) ===# tensor [3, 4, 5]
  (do slice [1.to 1] !x) ===# tensor []
  (do slice [1.to 2] !x) ===# tensor [4]
  (do slice [1.to 3] !x) ===# tensor [4, 5]
  (do slice [2.to 2] !x) ===# tensor []
  (do slice [2.to 3] !x) ===# tensor [5]

partial
sliceStaticMixed : Property
sliceStaticMixed = fixedProperty $ do
  let x = tensor {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do slice [0.to 1] !x) ===# tensor [[3, 4, 5]]
  (do slice [1.to 1] !x) ===# tensor []
  (do slice [all, 2.to 2] !x) ===# tensor [[], []]
  (do slice [all, 1.to 3] !x) ===# tensor [[4, 5], [7, 8]]
  (do slice [at 0, 2.to 2] !x) ===# tensor []
  (do slice [at 0, 1.to 3] !x) ===# tensor [4, 5]
  (do slice [at 1, 2.to 2] !x) ===# tensor []
  (do slice [at 1, 1.to 3] !x) ===# tensor [7, 8]
  (do slice [0.to 1, at 0] !x) ===# tensor [3]
  (do slice [0.to 1, at 1] !x) ===# tensor [4]
  (do slice [0.to 1, at 2] !x) ===# tensor [5]
  (do slice [1.to 2, at 0] !x) ===# tensor [6]
  (do slice [1.to 2, at 1] !x) ===# tensor [7]
  (do slice [1.to 2, at 2] !x) ===# tensor [8]

-- idris is having a hard time inferring types
u64 : Nat -> Graph $ Tensor [] U64
u64 = tensor . Scalar

partial
sliceDynamicIndex : Property
sliceDynamicIndex = fixedProperty $ do
  let x = tensor {dtype=S32} [3, 4, 5]
  (do slice [at (!(u64 0))] !x) ===# tensor 3
  (do slice [at (!(u64 1))] !x) ===# tensor 4
  (do slice [at (!(u64 2))] !x) ===# tensor 5
  (do slice [at (!(u64 3))] !x) ===# tensor 5
  (do slice [at (!(u64 5))] !x) ===# tensor 5

  let x = tensor {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do slice [at (!(u64 0))] !x) ===# tensor [3, 4, 5]
  (do slice [at (!(u64 1))] !x) ===# tensor [6, 7, 8]
  (do slice [at (!(u64 2))] !x) ===# tensor [6, 7, 8]
  (do slice [at (!(u64 4))] !x) ===# tensor [6, 7, 8]

partial
sliceDynamicSlice : Property
sliceDynamicSlice = fixedProperty $ do
  let x = tensor {dtype=S32} [3, 4, 5]
  (do slice [(!(u64 0)).size 0] !x) ===# tensor []
  (do slice [(!(u64 0)).size 1] !x) ===# tensor [3]
  (do slice [(!(u64 0)).size 2] !x) ===# tensor [3, 4]
  (do slice [(!(u64 0)).size 3] !x) ===# tensor [3, 4, 5]
  (do slice [(!(u64 1)).size 0] !x) ===# tensor []
  (do slice [(!(u64 1)).size 1] !x) ===# tensor [4]
  (do slice [(!(u64 1)).size 2] !x) ===# tensor [4, 5]
  (do slice [(!(u64 1)).size 3] !x) ===# tensor [3, 4, 5]
  (do slice [(!(u64 2)).size 0] !x) ===# tensor []
  (do slice [(!(u64 2)).size 1] !x) ===# tensor [5]
  (do slice [(!(u64 3)).size 0] !x) ===# tensor []
  (do slice [(!(u64 3)).size 1] !x) ===# tensor [5]
  (do slice [(!(u64 3)).size 3] !x) ===# tensor [3, 4, 5]
  (do slice [(!(u64 5)).size 0] !x) ===# tensor []
  (do slice [(!(u64 5)).size 1] !x) ===# tensor [5]
  (do slice [(!(u64 5)).size 3] !x) ===# tensor [3, 4, 5]

  let x = tensor {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do slice [(!(u64 0)).size 1] !x) ===# tensor [[3, 4, 5]]
  (do slice [(!(u64 1)).size 0] !x) ===# tensor []
  (do slice [(!(u64 2)).size 0] !x) ===# tensor []
  (do slice [(!(u64 2)).size 1] !x) ===# tensor [[6, 7, 8]]
  (do slice [(!(u64 4)).size 0] !x) ===# tensor []
  (do slice [(!(u64 4)).size 1] !x) ===# tensor [[6, 7, 8]]

partial
sliceMixed : Property
sliceMixed = fixedProperty $ do
  let x = tensor {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do slice [all, (!(u64 2)).size 0] !x) ===# tensor [[], []]
  (do slice [all, (!(u64 1)).size 2] !x) ===# tensor [[4, 5], [7, 8]]
  (do slice [all, (!(u64 3)).size 0] !x) ===# tensor [[], []]
  (do slice [all, (!(u64 3)).size 2] !x) ===# tensor [[4, 5], [7, 8]]
  (do slice [all, (!(u64 5)).size 0] !x) ===# tensor [[], []]
  (do slice [all, (!(u64 5)).size 2] !x) ===# tensor [[4, 5], [7, 8]]
  (do slice [at 0, (!(u64 2)).size 0] !x) ===# tensor []
  (do slice [at 0, (!(u64 1)).size 2] !x) ===# tensor [4, 5]
  (do slice [at 1, (!(u64 2)).size 0] !x) ===# tensor []
  (do slice [at 1, (!(u64 1)).size 2] !x) ===# tensor [7, 8]
  (do slice [at 1, (!(u64 3)).size 0] !x) ===# tensor []
  (do slice [at 1, (!(u64 3)).size 2] !x) ===# tensor [7, 8]
  (do slice [at 1, (!(u64 5)).size 0] !x) ===# tensor []
  (do slice [at 1, (!(u64 5)).size 2] !x) ===# tensor [7, 8]
  (do slice [(!(u64 0)).size 1, at 0] !x) ===# tensor [3]
  (do slice [(!(u64 0)).size 1, at 1] !x) ===# tensor [4]
  (do slice [(!(u64 0)).size 1, at 2] !x) ===# tensor [5]
  (do slice [(!(u64 1)).size 1, at 0] !x) ===# tensor [6]
  (do slice [(!(u64 1)).size 1, at 1] !x) ===# tensor [7]
  (do slice [(!(u64 1)).size 1, at 2] !x) ===# tensor [8]
  (do slice [(!(u64 2)).size 1, at 0] !x) ===# tensor [6]
  (do slice [(!(u64 2)).size 1, at 1] !x) ===# tensor [7]
  (do slice [(!(u64 2)).size 1, at 2] !x) ===# tensor [8]
  (do slice [(!(u64 4)).size 1, at 0] !x) ===# tensor [6]
  (do slice [(!(u64 4)).size 1, at 1] !x) ===# tensor [7]
  (do slice [(!(u64 4)).size 1, at 2] !x) ===# tensor [8]

  let x : Array [60] Int32 = fromList [0..59]
      x = (do reshape {to=[2, 5, 3, 2]} !(tensor {shape=[60]} {dtype=S32} $ cast x))

  let idx = tensor {dtype=U64} 0
      start = tensor {dtype=U64} 1
  (do slice [at 1, 2.to 5, (!start).size 2, at !idx] !x) ===# tensor [[44, 46], [50, 52], [56, 58]]

index : (idx : Nat) -> {auto 0 inDim : LT idx n} -> Literal [n] a -> Literal [] a
index {inDim = (LTESucc _)} 0 (y :: _) = y
index {inDim = (LTESucc _)} (S k) (_ :: xs) = index k xs

partial
sliceForVariableIndex : Property
sliceForVariableIndex = property $ do
  idx <- forAll dims
  rem <- forAll dims
  lit <- forAll (literal [idx + S rem] nats)
  let x = tensor {dtype=U32} lit
  index @{inDim} idx lit === unsafeEval (do slice [at @{inDim} idx] !x)

  where
  %hint
  inDim : {idx, rem : _} -> LTE (S idx) (idx + S rem)
  inDim {idx = 0} = LTESucc LTEZero
  inDim {idx = (S k)} = LTESucc inDim

export partial
all : List (PropertyName, Property)
all = [
{-
      ("MultiSlice.slice", MultiSlice.slice)
    , ("slice for static index", sliceStaticIndex)
    , ("slice for static slice", sliceStaticSlice)
    , ("slice for dynamic index", sliceDynamicIndex)
    , ("slice for dynamic slice", sliceDynamicSlice)
    , ("slice for static index and slice", sliceStaticMixed)
    , ("slice for mixed static and dynamic index and slice", sliceMixed)
    , ("slice for variable index", sliceForVariableIndex)
    -}
  ]
