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
module Unit.TestTensor.Structure

import Data.Nat
import Data.Vect
import System

import Device
import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

partial
reshape : Device => Property
reshape = fixedProperty $ do
  reshape 3 ===# tensor {dtype = S32} [3]

  let x = tensor {dtype = S32} [3, 4, 5]
      flipped = tensor [[3], [4], [5]]
  reshape x ===# flipped

  let x = tensor {dtype = S32} [[3, 4, 5], [6, 7, 8]]
      flipped = tensor [[3, 4], [5, 6], [7, 8]]
  reshape x ===# flipped

  let withExtraDim = tensor {dtype = S32} [[[3, 4, 5]], [[6, 7, 8]]]
  reshape x ===# withExtraDim

  let flattened = tensor {dtype = S32} [3, 4, 5, 6, 7, 8]
  reshape x ===# flattened

partial
expand : Device => Property
expand = fixedProperty $ do
  expand 0 3 ===# tensor {dtype = S32} [3]

  let x = tensor {dtype = S32} [[3, 4, 5], [6, 7, 8]]
      withExtraDim = tensor [[[3, 4, 5]], [[6, 7, 8]]]
  expand 1 x ===# withExtraDim

partial
broadcast : Device => Property
broadcast = fixedProperty $ do
  broadcast {to = []} {dtype = S32} 7 ===# 7
  broadcast {to = [1]} {dtype = S32} 7 ===# tensor [7]
  broadcast {to = [2, 3]} {dtype = S32} 7 ===# tensor [[7, 7, 7], [7, 7, 7]]
  broadcast {to = [1, 1, 1]} {dtype = S32} 7 ===# tensor [[[7]]]
  broadcast {to = [0]} {dtype = S32} 7 ===# tensor []

  let x = tensor {dtype = S32} [7]
  broadcast {to = [1]} x ===# tensor [7]

  let x = tensor {dtype = S32} [7]
  broadcast {to = [3]} x ===# tensor [7, 7, 7]

  let x = tensor {dtype = S32} [7]
  broadcast {to = [2, 3]} x ===# tensor [[7, 7, 7], [7, 7, 7]]

  let x = tensor {dtype = S32} [5, 7]
  broadcast {to = [2, 0]} x ===# tensor [[], []]

  let x = tensor {dtype = S32} [5, 7]
  broadcast {to = [3, 2]} x ===# tensor [[5, 7], [5, 7], [5, 7]]

  let x = tensor {dtype = S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to = [2, 3]} x ===# tensor [[2, 3, 5], [7, 11, 13]]

  let x = tensor {dtype = S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to = [2, 0]} x ===# tensor [[], []]

  let x = tensor {dtype = S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to = [0, 3]} x ===# tensor []

  let x = tensor {dtype = S32} [[2, 3, 5], [7, 11, 13]]
      expected = tensor [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  broadcast {to = [2, 2, 3]} x ===# expected

  let x = tensor {dtype = S32} [[[2, 3, 5]], [[7, 11, 13]]]
      expected = tensor [
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ],
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ]
      ]
  broadcast {to = [2, 2, 5, 3]} x ===# expected

partial
triangle : Device => Property
triangle = fixedProperty $ do
  let x = tensor {dtype = S32} []
  triangle Upper x ===# tensor []
  triangle Lower x ===# tensor []

  let x = tensor {dtype = S32} [[3]]
  triangle Upper x ===# tensor [[3]]
  triangle Lower x ===# tensor [[3]]

  let x = tensor {dtype = S32} [[1, 2], [3, 4]]
  triangle Upper x ===# tensor [[1, 2], [0, 4]]
  triangle Lower x ===# tensor [[1, 0], [3, 4]]

  let x = tensor {dtype = S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  triangle Upper x ===# tensor [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  triangle Lower x ===# tensor [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

partial
diag : Device => Property
diag = fixedProperty $ do
  let x = tensor {dtype = S32} []
  diag x ===# tensor []

  let x = tensor {dtype = S32} [[3]]
  diag x ===# tensor [3]

  let x = tensor {dtype = S32} [[1, 2], [3, 4]]
  diag x ===# tensor [1, 4]

partial
concat : Device => Property
concat = fixedProperty $ do
  let vector = tensor {dtype = S32} [3, 4, 5]

  let l = tensor {shape = [0]} []
      r = tensor [3, 4, 5]
  concat 0 l r ===# vector

  let l = tensor [3]
      r = tensor [4, 5]
  concat 0 l r ===# vector

  let l = tensor [3, 4]
      r = tensor [5]
  concat 0 l r ===# vector

  let l = tensor [3, 4, 5]
      r = tensor {shape = [0]} []
  concat 0 l r ===# vector

  let arr = tensor {dtype = S32} [[3, 4, 5], [6, 7, 8]]

  let l = tensor {shape = [0, 3]} []
      r = tensor [[3, 4, 5], [6, 7, 8]]
  concat 0 l r ===# arr

  let l = tensor [[3, 4, 5]]
      r = tensor [[6, 7, 8]]
  concat 0 l r ===# arr

  let l = tensor [[3, 4, 5], [6, 7, 8]]
      r = tensor {shape = [0, 3]} []
  concat 0 l r ===# arr

  let l = tensor {shape = [2, 0]} [[], []]
      r = tensor [[3, 4, 5], [6, 7, 8]]
  concat 1 l r ===# arr

  let l = tensor [[3], [6]]
      r = tensor [[4, 5], [7, 8]]
  concat 1 l r ===# arr

  let l = tensor [[3, 4], [6, 7]]
      r = tensor [[5], [8]]
  concat 1 l r ===# arr

  let l = tensor [[3, 4, 5], [6, 7, 8]]
      r = tensor {shape = [2, 0]} [[], []]
  concat 1 l r ===# arr

dimBroadcastable : List (a ** b ** DimBroadcastable a b)
dimBroadcastable = [
  (0 ** 0 ** Same),
  (1 ** 1 ** Same),
  (3 ** 3 ** Same),
  (1 ** 0 ** Stack),
  (1 ** 1 ** Stack),
  (1 ** 3 ** Stack),
  (0 ** 0 ** Zero),
  (1 ** 0 ** Zero),
  (3 ** 0 ** Zero)
]

broadcastable : List (from : Shape ** to : Shape ** Broadcastable from to)
broadcastable = [
  ([] ** [] ** Same),
  ([3, 2, 5] ** [3, 2, 5] ** Same),
  ([] ** [3, 2, 5] ** Nest $ Nest $ Nest Same),
  ([3, 1, 5] ** [3, 7, 5] ** Match $ Match Same),
  ([3, 2, 5] ** [1, 3, 2, 5] ** Nest Same),
  ([3, 2, 5] ** [7, 3, 2, 5] ** Nest Same)
]

broadcastableCannotReduceRank0 : Broadcastable [5] [] -> Void
broadcastableCannotReduceRank0 _ impossible

broadcastableCannotReduceRank1 : Broadcastable [3, 2, 5] [] -> Void
broadcastableCannotReduceRank1 _ impossible

broadcastableCannotStackDimensionGtOne : Broadcastable [3, 2] [3, 7] -> Void
broadcastableCannotStackDimensionGtOne (Match Same) impossible
broadcastableCannotStackDimensionGtOne (Nest Same) impossible

partial
squeeze : Device => Property
squeeze = fixedProperty $ do
  let x = tensor {dtype = S32} [[3]]
  squeeze x ===# 3

  let x = tensor {dtype = S32} [[[3, 4, 5]], [[6, 7, 8]]]
  squeeze x ===# x

  let squeezed = tensor {dtype = S32} [[3, 4, 5], [6, 7, 8]]
  squeeze x ===# squeezed

  let x = fill {shape = [1, 3, 1, 1, 2, 5, 1]} {dtype = S32} 0
  squeeze x ===# fill {shape = [3, 2, 5]} {dtype = S32} 0

squeezableCannotRemoveNonOnes : Squeezable [1, 2] [] -> Void
squeezableCannotRemoveNonOnes (Nest _) impossible

partial
(.T) : Device => Property
(.T) = fixedProperty $ do
  (tensor {dtype = S32} []).T ===# tensor []
  (tensor {dtype = S32} [[3]]).T ===# tensor [[3]]

  let x = tensor {dtype = S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = tensor [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  x.T ===# expected

partial
transpose : Device => Property
transpose = fixedProperty $ do
  let x = tensor {dtype = S32} [[0, 1], [2, 3]]
  transpose [0, 1] x ===# x
  transpose [1, 0] x ===# tensor [[0, 2], [1, 3]]

  let x = tensor {dtype = S32}
        [[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],
         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]
  transpose [0, 2, 1] x ===# tensor
    [[[ 0,  4,  8],
      [ 1,  5,  9],
      [ 2,  6, 10],
      [ 3,  7, 11]],
     [[12, 16, 20],
      [13, 17, 21],
      [14, 18, 22],
      [15, 19, 23]]]
  transpose [2, 0, 1] x ===# tensor
    [[[ 0,  4,  8],
      [12, 16, 20]],
     [[ 1,  5,  9],
      [13, 17, 21]],
     [[ 2,  6, 10],
      [14, 18, 22]],
     [[ 3,  7, 11],
      [15, 19, 23]]]

  let x : Array [120] Int32 = fromList [0..119]
      x : Tensor [2, 3, 4, 5] S32 = reshape $ tensor {shape = [120]} (cast x)
  transpose [0, 1, 2, 3] x ===# x
  slice [all, at 1, at 0] (transpose [0, 2, 1, 3] x) ===# slice [all, at 0, at 1] x
  slice [at 2, at 4, at 0, at 1] (transpose [2, 3, 1, 0] x) ===# slice [at 1, at 0, at 2, at 4] x

partial
reverse : Device => Property
reverse = fixedProperty $ do
  let x = tensor {shape = [0]} {dtype = S32} []
  reverse [0] x ===# x

  let x = tensor {shape = [0, 3]} {dtype = S32} []
  reverse [0] x ===# x
  reverse [1] x ===# x
  reverse [0, 1] x ===# x

  let x = tensor {dtype = S32} [-2, 0, 1]
  reverse [0] x ===# tensor [1, 0, -2]

  let x = tensor {dtype = S32} [[0, 1, 2], [3, 4, 5]]
  reverse [0] x ===# tensor [[3, 4, 5], [0, 1, 2]]
  reverse [1] x ===# tensor [[2, 1, 0], [5, 4, 3]]
  reverse [0, 1] x ===# tensor [[5, 4, 3], [2, 1, 0]]

  let x = tensor {dtype = S32} [
    [[[ 0,  1], [ 2,  3]], [[ 4,  5], [ 6,  7]], [[ 8,  9], [10, 11]]],
    [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]
  ]
  reverse [0, 3] x ===# tensor [
    [[[13, 12], [15, 14]], [[17, 16], [19, 18]], [[21, 20], [23, 22]]],
    [[[ 1,  0], [ 3,  2]], [[ 5,  4], [ 7,  6]], [[ 9,  8], [11, 10]]]
  ]

export partial
all : Device => List (PropertyName, Property)
all = [
      ("reshape", reshape)
    , ("concat", concat)
    , ("triangle", triangle)
    , ("diag", diag)
    , ("expand", expand)
    , ("broadcast", broadcast)
    , ("squeeze", squeeze)
    , ("(.T)", (.T))
    , ("transpose", transpose)
    , ("reverse", reverse)
  ]
