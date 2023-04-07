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

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

partial
reshape : Property
reshape = fixedProperty $ do
  (do reshape !3) ===# fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [3, 4, 5]
      flipped = fromLiteral [[3], [4], [5]]
  (do reshape !x) ===# flipped

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      flipped = fromLiteral [[3, 4], [5, 6], [7, 8]]
  (do reshape !x) ===# flipped

  let withExtraDim = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  (do reshape !x) ===# withExtraDim

  let flattened = fromLiteral {dtype=S32} [3, 4, 5, 6, 7, 8]
  (do reshape !x) ===# flattened

partial
expand : Property
expand = fixedProperty $ do
  (do expand 0 !3) ===# fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      withExtraDim = fromLiteral [[[3, 4, 5]], [[6, 7, 8]]]
  (do expand 1 !x) ===# withExtraDim

partial
broadcast : Property
broadcast = fixedProperty $ do
  (do broadcast {to=[]} {dtype=S32} !7) ===# 7
  (do broadcast {to=[1]} {dtype=S32} !7) ===# fromLiteral [7]
  (do broadcast {to=[2, 3]} {dtype=S32} !7) ===# fromLiteral [[7, 7, 7], [7, 7, 7]]
  (do broadcast {to=[1, 1, 1]} {dtype=S32} !7) ===# fromLiteral [[[7]]]
  (do broadcast {to=[0]} {dtype=S32} !7) ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [7]
  (do broadcast {to=[1]} !x) ===# fromLiteral [7]

  let x = fromLiteral {dtype=S32} [7]
  (do broadcast {to=[3]} !x) ===# fromLiteral [7, 7, 7]

  let x = fromLiteral {dtype=S32} [7]
  (do broadcast {to=[2, 3]} !x) ===# fromLiteral [[7, 7, 7], [7, 7, 7]]

  let x = fromLiteral {dtype=S32} [5, 7]
  (do broadcast {to=[2, 0]} !x) ===# fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [5, 7]
  (do broadcast {to=[3, 2]} !x) ===# fromLiteral [[5, 7], [5, 7], [5, 7]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  (do broadcast {to=[2, 3]} !x) ===# fromLiteral [[2, 3, 5], [7, 11, 13]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  (do broadcast {to=[2, 0]} !x) ===# fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  (do broadcast {to=[0, 3]} !x) ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
      expected = fromLiteral [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  (do broadcast {to=[2, 2, 3]} !x) ===# expected

  let x = fromLiteral {dtype=S32} [[[2, 3, 5]], [[7, 11, 13]]]
      expected = fromLiteral [
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ],
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ]
      ]
  (do broadcast {to=[2, 2, 5, 3]} !x) ===# expected

partial
triangle : Property
triangle = fixedProperty $ do
  let x = fromLiteral {dtype=S32} []
  (do triangle Upper !x) ===# fromLiteral []
  (do triangle Lower !x) ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  (do triangle Upper !x) ===# fromLiteral [[3]]
  (do triangle Lower !x) ===# fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  (do triangle Upper !x) ===# fromLiteral [[1, 2], [0, 4]]
  (do triangle Lower !x) ===# fromLiteral [[1, 0], [3, 4]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  (do triangle Upper !x) ===# fromLiteral [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  (do triangle Lower !x) ===# fromLiteral [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

partial
diag : Property
diag = fixedProperty $ do
  let x = fromLiteral {dtype=S32} []
  (do diag !x) ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  (do diag !x) ===# fromLiteral [3]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  (do diag !x) ===# fromLiteral [1, 4]

partial
concat : Property
concat = fixedProperty $ do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let l = fromLiteral {shape=[0]} []
      r = fromLiteral [3, 4, 5]
  (do concat 0 !l !r) ===# vector

  let l = fromLiteral [3]
      r = fromLiteral [4, 5]
  (do concat 0 !l !r) ===# vector

  let l = fromLiteral [3, 4]
      r = fromLiteral [5]
  (do concat 0 !l !r) ===# vector

  let l = fromLiteral [3, 4, 5]
      r = fromLiteral {shape=[0]} []
  (do concat 0 !l !r) ===# vector

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let l = fromLiteral {shape=[0, 3]} []
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  (do concat 0 !l !r) ===# arr

  let l = fromLiteral [[3, 4, 5]]
      r = fromLiteral [[6, 7, 8]]
  (do concat 0 !l !r) ===# arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[0, 3]} []
  (do concat 0 !l !r) ===# arr

  let l = fromLiteral {shape=[2, 0]} [[], []]
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  (do concat 1 !l !r) ===# arr

  let l = fromLiteral [[3], [6]]
      r = fromLiteral [[4, 5], [7, 8]]
  (do concat 1 !l !r) ===# arr

  let l = fromLiteral [[3, 4], [6, 7]]
      r = fromLiteral [[5], [8]]
  (do concat 1 !l !r) ===# arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[2, 0]} [[], []]
  (do concat 1 !l !r) ===# arr

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
squeeze : Property
squeeze = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[3]]
  (do squeeze !x) ===# 3

  let x = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  (do squeeze !x) ===# x

  let squeezed = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do squeeze !x) ===# squeezed

  let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
  (do squeeze !x) ===# fill {shape=[3, 2, 5]} {dtype=S32} 0

squeezableCannotRemoveNonOnes : Squeezable [1, 2] [] -> Void
squeezableCannotRemoveNonOnes (Nest _) impossible

partial
(.T) : Property
(.T) = fixedProperty $ do
  (do (fromLiteral {dtype=S32} []).T) ===# fromLiteral []
  (do (fromLiteral {dtype=S32} [[3]]).T) ===# fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = fromLiteral [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  x.T ===# expected

partial
transpose : Property
transpose = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[0, 1], [2, 3]]
  (do transpose [0, 1] !x) ===# x
  (do transpose [1, 0] !x) ===# fromLiteral [[0, 2], [1, 3]]

  let x = fromLiteral {dtype=S32}
        [[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],
         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]
  (do transpose [0, 2, 1] !x) ===# fromLiteral
    [[[ 0,  4,  8],
      [ 1,  5,  9],
      [ 2,  6, 10],
      [ 3,  7, 11]],
     [[12, 16, 20],
      [13, 17, 21],
      [14, 18, 22],
      [15, 19, 23]]]
  (do transpose [2, 0, 1] !x) ===# fromLiteral
    [[[ 0,  4,  8],
      [12, 16, 20]],
     [[ 1,  5,  9],
      [13, 17, 21]],
     [[ 2,  6, 10],
      [14, 18, 22]],
     [[ 3,  7, 11],
      [15, 19, 23]]]

  let x : Array [120] Int32 = fromList [0..119]
      x : Ref $ Tensor [2, 3, 4, 5] S32 = (do reshape !(fromLiteral {shape=[120]} (cast x)))
  (do transpose [0, 1, 2, 3] !x) ===# x
  (do slice [all, at 1, at 0] !(transpose [0, 2, 1, 3] !x)) ===# (do slice [all, at 0, at 1] !x)
  (do slice [at 2, at 4, at 0, at 1] !(transpose [2, 3, 1, 0] !x)) ===# (do slice [at 1, at 0, at 2, at 4] !x)

partial
reverse : Property
reverse = fixedProperty $ do
  let x = fromLiteral {shape=[0]} {dtype=S32} []
  (do reverse [0] !x) ===# x

  let x = fromLiteral {shape=[0, 3]} {dtype=S32} []
  (do reverse [0] !x) ===# x
  (do reverse [1] !x) ===# x
  (do reverse [0, 1] !x) ===# x

  let x = fromLiteral {dtype=S32} [-2, 0, 1]
  (do reverse [0] !x) ===# fromLiteral [1, 0, -2]

  let x = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
  (do reverse [0] !x) ===# fromLiteral [[3, 4, 5], [0, 1, 2]]
  (do reverse [1] !x) ===# fromLiteral [[2, 1, 0], [5, 4, 3]]
  (do reverse [0, 1] !x) ===# fromLiteral [[5, 4, 3], [2, 1, 0]]

  let x = fromLiteral {dtype=S32} [
    [[[ 0,  1], [ 2,  3]], [[ 4,  5], [ 6,  7]], [[ 8,  9], [10, 11]]],
    [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]
  ]
  (do reverse [0, 3] !x) ===# fromLiteral [
    [[[13, 12], [15, 14]], [[17, 16], [19, 18]], [[21, 20], [23, 22]]],
    [[[ 1,  0], [ 3,  2]], [[ 5,  4], [ 7,  6]], [[ 9,  8], [11, 10]]]
  ]

export partial
all : List (PropertyName, Property)
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
