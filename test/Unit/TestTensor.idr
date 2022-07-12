{--
Copyright 2021 Joel Berkeley

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
module Unit.TestTensor

import Control.Monad.State
import Data.List.Quantifiers
import Data.Nat
import Data.Vect
import System

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

covering
fromLiteralThenToLiteral : Property
fromLiteralThenToLiteral = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  x ==~ toLiteral (fromLiteral {dtype=F64} x)

  x <- forAll (literal shape ints)
  x === toLiteral (fromLiteral {dtype=S32} x)

  x <- forAll (literal shape nats)
  x === toLiteral (fromLiteral {dtype=U32} x)

  x <- forAll (literal shape nats)
  x === toLiteral (fromLiteral {dtype=U64} x)

  x <- forAll (literal shape bool)
  x === toLiteral (fromLiteral {dtype=PRED} x)

canConvertAtXlaNumericBounds : Property
canConvertAtXlaNumericBounds = fixedProperty $ do
  let f64lim : Literal [] Double = 1.7976931348623157e308
      min' : Tensor [] F64 = Bounded.min @{Finite}
      max' : Tensor [] F64 = Bounded.max @{Finite}
  toLiteral min' === -f64lim
  toLiteral max' === f64lim
  toLiteral (fromLiteral (-f64lim) == min') === True
  toLiteral (fromLiteral f64lim == max') === True

  let s32min : Literal [] Int = -2147483648
      s32max : Literal [] Int = 2147483647
      min' : Tensor [] S32 = Bounded.min @{Finite}
      max' : Tensor [] S32 = Bounded.max @{Finite}
  toLiteral min' === s32min
  toLiteral max' === s32max
  toLiteral (fromLiteral (-s32min) == min') === True
  toLiteral (fromLiteral s32max == max') === True

  let u32min : Literal [] Nat = 0
      u32max : Literal [] Nat = 4294967295
      min' : Tensor [] U32 = Bounded.min @{Finite}
      max' : Tensor [] U32 = Bounded.max @{Finite}
  toLiteral min' === u32min
  toLiteral max' === u32max
  toLiteral (fromLiteral u32min == min') === True
  toLiteral (fromLiteral u32max == max') === True

  let u64min : Literal [] Nat = 0
      u64max : Literal [] Nat = 18446744073709551615
      min' : Tensor [] U64 = Bounded.min @{Finite}
      max' : Tensor [] U64 = Bounded.max @{Finite}
  toLiteral min' === u64min
  toLiteral max' === u64max
  toLiteral (fromLiteral u64min == min') === True
  toLiteral (fromLiteral u64max == max') === True

show : Property
show = fixedProperty $ do
  let x : Tensor [] S32 = 1
  show x === "constant, shape=[], metadata={:0}"

  let x : Tensor [] S32 = 1 + 2
  show x ===
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = fromLiteral {dtype=F64} [1.3, 2.0, -0.4]
  show x === "constant, shape=[3], metadata={:0}"

covering
cast : Property
cast = property $ do
  shape <- forAll shapes

  lit <- forAll (literal shape nats)
  let x : Tensor shape F64 = cast (fromLiteral {dtype=U32} lit)
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape nats)
  let x : Tensor shape F64 = cast (fromLiteral {dtype=U64} lit)
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape ints)
  let x : Tensor shape F64 = cast (fromLiteral {dtype=S32} lit)
  x ===# fromLiteral (map (cast {to=Double}) lit)

reshape : Property
reshape = fixedProperty $ do
  reshape 3 ===# fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [3, 4, 5]
      flipped = fromLiteral [[3], [4], [5]]
  reshape x ===# flipped

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      flipped = fromLiteral [[3, 4], [5, 6], [7, 8]]
  reshape x ===# flipped

  let withExtraDim = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  reshape x ===# withExtraDim

  let flattened = fromLiteral {dtype=S32} [3, 4, 5, 6, 7, 8]
  reshape x ===# flattened

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

slice : Property
slice = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  slice [at 0] x ===# fromLiteral 3
  slice [at 1] x ===# fromLiteral 4
  slice [at 2] x ===# fromLiteral 5
  slice [0.to 0] x ===# fromLiteral []
  slice [0.to 1] x ===# fromLiteral [3]
  slice [0.to 2] x ===# fromLiteral [3, 4]
  slice [0.to 3] x ===# fromLiteral [3, 4, 5]
  slice [1.to 1] x ===# fromLiteral []
  slice [1.to 2] x ===# fromLiteral [4]
  slice [1.to 3] x ===# fromLiteral [4, 5]
  slice [2.to 2] x ===# fromLiteral []
  slice [2.to 3] x ===# fromLiteral [5]

  let idx : Nat
      idx = 2

  slice [at idx] x ===# fromLiteral 5

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  slice [0.to 1] x ===# fromLiteral [[3, 4, 5]]
  slice [1.to 1] x ===# fromLiteral []
  slice [all, 2.to 2] x ===# fromLiteral [[], []]
  slice [all, 1.to 3] x ===# fromLiteral [[4, 5], [7, 8]]
  slice [at 0, 2.to 2] x ===# fromLiteral []
  slice [at 0, 1.to 3] x ===# fromLiteral [4, 5]
  slice [at 1, 2.to 2] x ===# fromLiteral []
  slice [at 1, 1.to 3] x ===# fromLiteral [7, 8]
  slice [0.to 1, at 0] x ===# fromLiteral [3]
  slice [0.to 1, at 1] x ===# fromLiteral [4]
  slice [0.to 1, at 2] x ===# fromLiteral [5]
  slice [1.to 2, at 0] x ===# fromLiteral [6]
  slice [1.to 2, at 1] x ===# fromLiteral [7]
  slice [1.to 2, at 2] x ===# fromLiteral [8]

  let x : Array [60] Int = fromList [0..59]
      x = reshape {to=[4, 5, 3]} (fromLiteral {shape=[60]} {dtype=S32} $ cast x)
  -- np.arange(60).reshape([4, 5, 3])[1:3, 0:4, 2]
  slice [1.to 3, 0.to 4, at 2] x ===# fromLiteral [[17, 20, 23, 26], [32, 35, 38, 41]]

index : (idx : Nat) -> {auto 0 inDim : LT idx n} -> Literal [n] a -> Literal [] a
index {inDim = (LTESucc _)} 0 (y :: _) = y
index {inDim = (LTESucc _)} (S k) (_ :: xs) = index k xs

covering
sliceForVariableIndex : Property
sliceForVariableIndex = property $ do
  idx <- forAll dims
  rem <- forAll dims
  lit <- forAll (literal [idx + S rem] nats)
  let x = fromLiteral {dtype=U32} lit
  index @{inDim} idx lit === toLiteral (slice [at @{inDim} idx] x)

  where
  %hint
  inDim : {idx, rem : _} -> LTE (S idx) (idx + S rem)
  inDim {idx = 0} = LTESucc LTEZero
  inDim {idx = (S k)} = LTESucc inDim

concat : Property
concat = fixedProperty $ do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let l = fromLiteral {shape=[0]} []
      r = fromLiteral [3, 4, 5]
  concat 0 l r ===# vector

  let l = fromLiteral [3]
      r = fromLiteral [4, 5]
  concat 0 l r ===# vector

  let l = fromLiteral [3, 4]
      r = fromLiteral [5]
  concat 0 l r ===# vector

  let l = fromLiteral [3, 4, 5]
      r = fromLiteral {shape=[0]} []
  concat 0 l r ===# vector

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let l = fromLiteral {shape=[0, 3]} []
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  concat 0 l r ===# arr

  let l = fromLiteral [[3, 4, 5]]
      r = fromLiteral [[6, 7, 8]]
  concat 0 l r ===# arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[0, 3]} []
  concat 0 l r ===# arr

  let l = fromLiteral {shape=[2, 0]} [[], []]
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  concat 1 l r ===# arr

  let l = fromLiteral [[3], [6]]
      r = fromLiteral [[4, 5], [7, 8]]
  concat 1 l r ===# arr

  let l = fromLiteral [[3, 4], [6, 7]]
      r = fromLiteral [[5], [8]]
  concat 1 l r ===# arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[2, 0]} [[], []]
  concat 1 l r ===# arr

diag : Property
diag = fixedProperty $ do
  let x = fromLiteral {dtype=S32} []
  diag x ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  diag x ===# fromLiteral [3]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  diag x ===# fromLiteral [1, 4]

triangle : Property
triangle = fixedProperty $ do
  let x = fromLiteral {dtype=S32} []
  triangle Upper x ===# fromLiteral []
  triangle Lower x ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  triangle Upper x ===# fromLiteral [[3]]
  triangle Lower x ===# fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  triangle Upper x ===# fromLiteral [[1, 2], [0, 4]]
  triangle Lower x ===# fromLiteral [[1, 0], [3, 4]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  triangle Upper x ===# fromLiteral [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  triangle Lower x ===# fromLiteral [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

identity : Property
identity = fixedProperty $ do
  identity ===# fromLiteral {dtype=S32} []
  identity ===# fromLiteral {dtype=S32} [[1]]
  identity ===# fromLiteral {dtype=S32} [[1, 0], [0, 1]]
  identity ===# fromLiteral {dtype=S32} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

  identity ===# fromLiteral {dtype=F64} []
  identity ===# fromLiteral {dtype=F64} [[1.0]]
  identity ===# fromLiteral {dtype=F64} [[1.0, 0.0], [0.0, 1.0]]
  identity ===# fromLiteral {dtype=F64} [
      [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    ]

expand : Property
expand = fixedProperty $ do
  expand 0 3 ===# fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      withExtraDim = fromLiteral [[[3, 4, 5]], [[6, 7, 8]]]
  expand 1 x ===# withExtraDim

broadcast : Property
broadcast = fixedProperty $ do
  broadcast {to=[]} {dtype=S32} 7 ===# 7
  broadcast {to=[1]} {dtype=S32} 7 ===# fromLiteral [7]
  broadcast {to=[2, 3]} 7 ===# fromLiteral [[7, 7, 7], [7, 7, 7]]
  broadcast {to=[1, 1, 1]} {dtype=S32} 7 ===# fromLiteral [[[7]]]
  broadcast {to=[0]} 7 ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[1]} x ===# fromLiteral [7]

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[3]} x ===# fromLiteral [7, 7, 7]

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[2, 3]} x ===# fromLiteral [[7, 7, 7], [7, 7, 7]]

  let x = fromLiteral {dtype=S32} [5, 7]
  broadcast {to=[2, 0]} x ===# fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [5, 7]
  broadcast {to=[3, 2]} x ===# fromLiteral [[5, 7], [5, 7], [5, 7]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[2, 3]} x ===# fromLiteral [[2, 3, 5], [7, 11, 13]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[2, 0]} x ===# fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[0, 3]} x ===# fromLiteral []

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
      expected = fromLiteral [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  broadcast {to=[2, 2, 3]} x ===# expected

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
  broadcast {to=[2, 2, 5, 3]} x ===# expected

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

squeeze : Property
squeeze = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[3]]
  squeeze x ===# 3

  let x = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  squeeze x ===# x

  let squeezed = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  squeeze x ===# squeezed

  let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
  squeeze x ===# fill {shape=[3, 2, 5]} {dtype=S32} 0

squeezableCannotRemoveNonOnes : Squeezable [1, 2] [] -> Void
squeezableCannotRemoveNonOnes (Nest _) impossible

(.T) : Property
(.T) = fixedProperty $ do
  (fromLiteral {dtype=S32} []).T ===# fromLiteral []
  (fromLiteral {dtype=S32} [[3]]).T ===# fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = fromLiteral [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  x.T ===# expected

transpose : Property
transpose = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[0, 1], [2, 3]]
  transpose [0, 1] x ===# x
  transpose [1, 0] x ===# fromLiteral [[0, 2], [1, 3]]

  let x = fromLiteral {dtype=S32}
        [[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],
         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]
  transpose [0, 2, 1] x ===# fromLiteral
    [[[ 0,  4,  8],
      [ 1,  5,  9],
      [ 2,  6, 10],
      [ 3,  7, 11]],
     [[12, 16, 20],
      [13, 17, 21],
      [14, 18, 22],
      [15, 19, 23]]]
  transpose [2, 0, 1] x ===# fromLiteral
    [[[ 0,  4,  8],
      [12, 16, 20]],
     [[ 1,  5,  9],
      [13, 17, 21]],
     [[ 2,  6, 10],
      [14, 18, 22]],
     [[ 3,  7, 11],
      [15, 19, 23]]]

  let x : Array [120] Int = fromList [0..119]
      x : Tensor [2, 3, 4, 5] S32 = reshape $ fromLiteral {shape=[120]} (cast x)
  transpose [0, 1, 2, 3] x ===# x
  slice [all, at 1, at 0] (transpose [0, 2, 1, 3] x) ===# slice [all, at 0, at 1] x
  slice [at 2, at 4, at 0, at 1] (transpose [2, 3, 1, 0] x) ===# slice [at 1, at 0, at 2, at 4] x

covering
mapResult : Property
mapResult = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral x
  map (1.0 /) x ==~ toLiteral (map (1.0 /) x')

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
  map (+ 1) x === toLiteral (map (+ 1) x')

mapNonTrivial : Property
mapNonTrivial = fixedProperty $ do
  map {a=S32} (\x => x + x) 1 ===# 2
  map {a=S32} (\_ => 2) 1 ===# 2
  map {a=S32} (map (+ 1)) 1 ===# 2

covering
map2Result : Property
map2Result = fixedProperty $ do
  shape <- forAll shapes

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x + y |] === toLiteral (map2 Tensor.(+) x' y')

  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x + y |] ==~ toLiteral (map2 Tensor.(+) x' y')

map2ResultWithReusedFnArgs : Property
map2ResultWithReusedFnArgs = fixedProperty $ do
  map2 (\x, y => x + x + y + y) 1 2 ===# 6

reduce : Property
reduce = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  reduce @{Sum} [1] x ===# fromLiteral [6, -6]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-2, -3, -4]]
  reduce @{Sum} [0, 1] x ===# fromLiteral (-3)

  let x = fromLiteral {dtype=S32} [[[1], [2], [3]], [[-2], [-3], [-4]]]
  reduce @{Sum} [0, 1] x ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3]], [[-2, -3, -4]]]
  reduce @{Sum} [0, 2] x ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3], [-2, -3, -4]]]
  reduce @{Sum} [1, 2] x ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3], [4, 5, 6]], [[-2, -3, -4], [-6, -7, -8]]]
  reduce @{Sum} [0, 2] x ===# fromLiteral [-3, -6]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  reduce @{Sum} [0] x ===# fromLiteral [0, 0, 0]

  let x = fromLiteral {dtype=PRED} [[True, False, True], [True, False, False]]
  reduce @{All} [1] x ===# fromLiteral [False, False]

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

covering
sort : Property
sort = withTests 20 . property $ do
  d <- forAll dims
  dd <- forAll dims
  ddd <- forAll dims

  x <- forAll (literal [S d] ints)
  let x = fromLiteral {dtype=S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] sorted
      tail = slice [1.to (S d)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  x <- forAll (literal [S d, S dd] ints)
  let x = fromLiteral {dtype=S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] sorted
      tail = slice [1.to (S d)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = sort (<) 1 x
      init = slice [all, 0.to dd] sorted
      tail = slice [all, 1.to (S dd)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  x <- forAll (literal [S d, S dd, S ddd] ints)
  let x = fromLiteral {dtype=S32} x

  let sorted = sort (<) 0 x
      init = slice [0.to d] sorted
      tail = slice [1.to (S d)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = sort (<) 1 x
      init = slice [all, 0.to dd] sorted
      tail = slice [all, 1.to (S dd)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = sort (<) 2 x
      init = slice [all, all, 0.to ddd] sorted
      tail = slice [all, all, 1.to (S ddd)] sorted
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  where
  %hint
  lteSucc : {n : _} -> LTE n (S n)
  lteSucc = lteSuccRight (reflexive {ty=Nat})

  %hint
  reflex : {n : _} -> LTE n n
  reflex = reflexive {ty=Nat}

sortWithEmptyAxis : Property
sortWithEmptyAxis = fixedProperty $ do
  let x = fromLiteral {shape=[0, 2, 3]} {dtype=S32} []
  sort (<) 0 x ===# x

  let x = fromLiteral {shape=[0, 2, 3]} {dtype=S32} []
  sort (<) 1 x ===# x

  let x = fromLiteral {shape=[2, 0, 3]} {dtype=S32} [[], []]
  sort (<) 0 x ===# x

  let x = fromLiteral {shape=[2, 0, 3]} {dtype=S32} [[], []]
  sort (<) 1 x ===# x

sortWithRepeatedElements : Property
sortWithRepeatedElements = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [1, 3, 4, 3, 2]
  sort (<) 0 x ===# fromLiteral [1, 2, 3, 3, 4]

  let x = fromLiteral {dtype=S32} [[1, 4, 4], [3, 2, 5]]
  sort (<) 0 x ===# fromLiteral [[1, 2, 4], [3, 4, 5]]
  sort (<) 1 x ===# fromLiteral [[1, 4, 4], [2, 3, 5]]

reverse : Property
reverse = fixedProperty $ do
  let x = fromLiteral {shape=[0]} {dtype=S32} []
  reverse [0] x ===# x

  let x = fromLiteral {shape=[0, 3]} {dtype=S32} []
  reverse [0] x ===# x
  reverse [1] x ===# x
  reverse [0, 1] x ===# x

  let x = fromLiteral {dtype=S32} [-2, 0, 1]
  reverse [0] x ===# fromLiteral [1, 0, -2]

  let x = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
  reverse [0] x ===# fromLiteral [[3, 4, 5], [0, 1, 2]]
  reverse [1] x ===# fromLiteral [[2, 1, 0], [5, 4, 3]]
  reverse [0, 1] x ===# fromLiteral [[5, 4, 3], [2, 1, 0]]

  let x = fromLiteral {dtype=S32} [
    [[[ 0,  1], [ 2,  3]], [[ 4,  5], [ 6,  7]], [[ 8,  9], [10, 11]]],
    [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]
  ]
  reverse [0, 3] x ===# fromLiteral [
    [[[13, 12], [15, 14]], [[17, 16], [19, 18]], [[21, 20], [23, 22]]],
    [[[ 1,  0], [ 3,  2]], [[ 5,  4], [ 7,  6]], [[ 9,  8], [11, 10]]]
  ]

namespace Vector
  export
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [-2, 0, 1]
        r = fromLiteral {dtype=S32} [3, 1, 2]
    l @@ r ===# -4

namespace Matrix
  export
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [3, 3, -1]
    l @@ r ===# fromLiteral [-7, 8]

    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    l @@ r ===# fromLiteral [[ -7,  -2], [  8, -11]]

namespace S32
  export covering
  testElementwiseUnary :
    (Int -> Int) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32) ->
    Property
  testElementwiseUnary fInt fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape ints)
    let x' = fromLiteral x
    [| fInt x |] === toLiteral (fTensor x')

namespace F64
  export covering
  testElementwiseUnary :
    (Double -> Double) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64) ->
    Property
  testElementwiseUnary fDouble fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape doubles)
    let x' = fromLiteral x
    [| fDouble x |] ==~ toLiteral (fTensor x')

namespace PRED
  export covering
  testElementwiseUnary :
    (Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseUnary fBool fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape bool)
    let x' = fromLiteral x
    [| fBool x |] === toLiteral (fTensor x')

covering
testElementwiseUnaryCases : List (PropertyName, Property)
testElementwiseUnaryCases = [
    ("negate S32", S32.testElementwiseUnary negate negate),
    ("negate F64", F64.testElementwiseUnary negate negate),
    ("recip", F64.testElementwiseUnary recip recip),
    ("abs S32", S32.testElementwiseUnary abs abs),
    ("abs F64", F64.testElementwiseUnary abs abs),
    ("exp", F64.testElementwiseUnary exp exp),
    ("ceil", F64.testElementwiseUnary ceiling ceil),
    ("floor", F64.testElementwiseUnary floor floor),
    ("log", F64.testElementwiseUnary log log),
    ("logistic", F64.testElementwiseUnary (\x => 1 / (1 + exp (-x))) logistic),
    ("sin", F64.testElementwiseUnary sin sin),
    ("cos", F64.testElementwiseUnary cos cos),
    ("tan", F64.testElementwiseUnary tan tan),
    ("asin", F64.testElementwiseUnary asin asin),
    ("acos", F64.testElementwiseUnary acos acos),
    ("atan", F64.testElementwiseUnary atan atan),
    ("sinh", F64.testElementwiseUnary sinh sinh),
    ("cosh", F64.testElementwiseUnary cosh cosh),
    ("tanh", F64.testElementwiseUnary tanh' tanh),
    ("asinh", F64.testElementwiseUnary asinh Tensor.asinh),
    ("acosh", F64.testElementwiseUnary acosh Tensor.acosh),
    ("atanh", F64.testElementwiseUnary atanh Tensor.atanh),
    ("sqrt", F64.testElementwiseUnary sqrt sqrt),
    ("square", F64.testElementwiseUnary (\x => x * x) square),
    ("not", PRED.testElementwiseUnary not not)
  ]

  where
  tanh' : Double -> Double
  tanh' x = let idrisResult = tanh x in
    if isNan idrisResult then
    if isNan x then idrisResult else
    if x < 0 then -1 else 1 else idrisResult

  asinh : Double -> Double
  asinh x = if x == -inf then -inf else log (x + sqrt (x * x + 1))

  acosh : Double -> Double
  acosh x = log (x + sqrt (x * x - 1))

  atanh : Double -> Double
  atanh x = log ((1 + x) / (1 - x)) / 2

namespace S32
  export covering
  testElementwiseBinary :
    (Int -> Int -> Int) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Tensor shape S32) ->
    Property
  testElementwiseBinary fInt fTensor = property $ do
    shape <- forAll shapes
    let ints = literal shape ints
    [x, y] <- forAll (np [ints, ints])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (fTensor x' y')

namespace F64
  export covering
  testElementwiseBinary :
    (Double -> Double -> Double) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Tensor shape F64) ->
    Property
  testElementwiseBinary fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] ==~ toLiteral (fTensor x' y')

namespace PRED
  export covering
  testElementwiseBinary :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseBinary fBool fTensor = property $ do
    shape <- forAll shapes
    let bools = literal shape bool
    [x, y] <- forAll (np [bools, bools])
    let x' = fromLiteral {dtype=PRED} x
        y' = fromLiteral {dtype=PRED} y
    [| fBool x y |] === toLiteral (fTensor x' y')

covering
testElementwiseBinaryCases : List (PropertyName, Property)
testElementwiseBinaryCases = [
    ("(+) F64", F64.testElementwiseBinary (+) (+)),
    ("(+) S32", S32.testElementwiseBinary (+) (+)),
    ("(-) F64", F64.testElementwiseBinary (-) (-)),
    ("(-) S32", S32.testElementwiseBinary (-) (-)),
    ("(*) F64", F64.testElementwiseBinary (*) (*)),
    ("(*) S32", S32.testElementwiseBinary (*) (*)),
    ("(/)", F64.testElementwiseBinary (/) (/)),
    -- ("pow", F64.testElementwiseBinary pow (^)),  bug in idris 0.5.1 for pow
    ("min S32", S32.testElementwiseBinary min min),
    ("max S32", S32.testElementwiseBinary max max),
    ("(&&)", PRED.testElementwiseBinary and (&&)),
    ("(||)", PRED.testElementwiseBinary or (||))
  ]

  where
  and : Bool -> Bool -> Bool
  and x y = x && y

  or : Bool -> Bool -> Bool
  or x y = x || y

covering
minF64 : Property
minF64 = property $ do
  shape <- forAll shapes
  -- XLA has a bug for nan values
  let doubles = literal shape doublesWithoutNan
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| min x y |] ==~ toLiteral (min x' y')

covering
maxF64 : Property
maxF64 = property $ do
  shape <- forAll shapes
  -- XLA has a bug for nan values
  let doubles = literal shape doublesWithoutNan
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| max x y |] ==~ toLiteral (max x' y')

covering
scalarMultiplication : Property
scalarMultiplication = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (scalar *) lit ==~ toLiteral (scalar' * lit')

covering
scalarDivision : Property
scalarDivision = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (/ scalar) lit ==~ toLiteral (lit' / scalar')

covering
neutralIsNeutralForSum : Property
neutralIsNeutralForSum = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right === x
  toLiteral left === x

covering
neutralIsNeutralForProd : Property
neutralIsNeutralForProd = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right === x
  toLiteral left === x

covering
neutralIsNeutralForAny : Property
neutralIsNeutralForAny = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (<+>) @{Any} x' (neutral @{Any})
      left = (<+>) @{Any} (neutral @{Any}) x'
  toLiteral right === x
  toLiteral left === x

covering
neutralIsNeutralForAll : Property
neutralIsNeutralForAll = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (<+>) @{All} x' (neutral @{All})
      left = (<+>) @{All} (neutral @{All}) x'
  toLiteral right === x
  toLiteral left === x

covering
neutralIsNeutralForMin : Property
neutralIsNeutralForMin = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Min} x' (neutral @{Min})
      left = (<+>) @{Min} (neutral @{Min}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

covering
neutralIsNeutralForMax : Property
neutralIsNeutralForMax = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Max} x' (neutral @{Max})
      left = (<+>) @{Max} (neutral @{Max}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

namespace S32
  export covering
  testElementwiseComparator :
    (Int -> Int -> Bool) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Tensor shape PRED) ->
    Property
  testElementwiseComparator fInt fTensor = property $ do
    shape <- forAll shapes
    let ints = literal shape ints
    [x, y] <- forAll (np [ints, ints])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (fTensor x' y')

namespace F64
  export covering
  testElementwiseComparator :
    (Double -> Double -> Bool) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Tensor shape PRED) ->
    Property
  testElementwiseComparator fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] === toLiteral (fTensor x' y')

namespace PRED
  export covering
  testElementwiseComparator :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseComparator = testElementwiseBinary

covering
testElementwiseComparatorCases : List (PropertyName, Property)
testElementwiseComparatorCases = [
    ("(==) F64", F64.testElementwiseComparator (==) (==)),
    ("(==) S32", S32.testElementwiseComparator (==) (==)),
    ("(==) PRED", PRED.testElementwiseComparator (==) (==)),
    ("(/=) F64", F64.testElementwiseComparator (/=) (/=)),
    ("(/=) S32", S32.testElementwiseComparator (/=) (/=)),
    ("(/=) PRED", PRED.testElementwiseComparator (/=) (/=)),
    ("(<) F64", F64.testElementwiseComparator (<) (<)),
    ("(<) S32", S32.testElementwiseComparator (<) (<)),
    ("(>) F64", F64.testElementwiseComparator (>) (>)),
    ("(>) S32", S32.testElementwiseComparator (>) (>)),
    ("(<=) F64", F64.testElementwiseComparator (<=) (<=)),
    ("(<=) S32", S32.testElementwiseComparator (<=) (<=)),
    ("(>=) F64", F64.testElementwiseComparator (>=) (>=)),
    ("(>=) S32", S32.testElementwiseComparator (>=) (>=))
  ]

select : Property
select = fixedProperty $ do
  let onTrue = fromLiteral {dtype=S32} 1
      onFalse = fromLiteral 0
  select (fromLiteral True) onTrue onFalse ===# onTrue
  select (fromLiteral False) onTrue onFalse ===# onFalse

  let pred = fromLiteral [[False, True, True], [True, False, False]]
      onTrue = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
      onFalse = fromLiteral [[6, 7, 8], [9, 10, 11]]
      expected = fromLiteral [[6, 1, 2], [3, 10, 11]]
  select pred onTrue onFalse ===# expected

condResultTrivialUsage : Property
condResultTrivialUsage = fixedProperty $ do
  let x = fromLiteral {dtype=S32} 0
  cond (fromLiteral True) (+ 1) x (\x => x - 1) x ===# 1

  let x = fromLiteral {dtype=S32} 0
  cond (fromLiteral False) (+ 1) x (\x => x - 1) x ===# -1

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  cond (fromLiteral True) (fromLiteral 5 *) x diag y ===# fromLiteral [10, 15]

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  cond (fromLiteral False) (fromLiteral 5 *) x diag y ===# fromLiteral [6, 9]

condResultWithReusedArgs : Property
condResultWithReusedArgs = fixedProperty $ do
  let x = fromLiteral {dtype=S32} 1
      y = fromLiteral {dtype=S32} 3
  cond (fromLiteral True) (\z => z + z) x (\z => z * z) y ===# 2
  cond (fromLiteral False) (\z => z + z) x (\z => z * z) y ===# 9

erf : Property
erf = fixedProperty $ do
  let x = fromLiteral [-1.5, -0.5, 0.5, 1.5]
      expected = fromLiteral [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  erf x ===# expected

cholesky : Property
cholesky = fixedProperty $ do
  let x = fromLiteral [[1.0, 0.0], [2.0, 0.0]]
      expected = fromLiteral [[nan, 0], [nan, nan]]
  cholesky x ===# expected

  -- example generated with tensorflow
  let x = fromLiteral [
              [ 2.236123  ,  0.70387983,  2.8447943 ],
              [ 0.7059226 ,  2.661426  , -0.8714733 ],
              [ 1.3730898 ,  1.4064665 ,  2.7474475 ]
            ]
      expected = fromLiteral [
              [1.4953672 , 0.0       , 0.0       ],
              [0.47207308, 1.5615932 , 0.0       ],
              [0.9182292 , 0.6230785 , 1.2312902 ]
            ]
  cholesky x ===# expected

triangularSolveResultAndInverse : Property
triangularSolveResultAndInverse = fixedProperty $ do
  let a = fromLiteral [
              [0.8578532 , 0.0       , 0.0       ],
              [0.2481904 , 0.9885198 , 0.0       ],
              [0.59390426, 0.14998078, 0.19468737]
            ]
      b = fromLiteral [
              [0.45312142, 0.37276268],
              [0.9210588 , 0.00647926],
              [0.7890165 , 0.77121615]
            ]
      actual = a |\ b
      expected = fromLiteral [
                    [ 0.52820396,  0.43452972],
                    [ 0.79913783, -0.10254406],
                    [ 1.8257918 ,  2.7147462 ]
                  ]
  actual ===# expected
  a @@ actual ===# b

  let actual = a.T \| b
      expected = fromLiteral [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  actual ===# expected
  a.T @@ actual ===# b

triangularSolveIgnoresOppositeElems : Property
triangularSolveIgnoresOppositeElems = fixedProperty $ do
  let a = fromLiteral [[1.0, 2.0], [3.0, 4.0]]
      aLower = fromLiteral [[1.0, 0.0], [3.0, 4.0]]
      b = fromLiteral [5.0, 6.0]
  a |\ b ===# aLower |\ b

  let aUpper = fromLiteral [[1.0, 2.0], [0.0, 4.0]]
  a \| b ===# aUpper \| b

trace : Property
trace = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[-1, 5], [1, 4]]
  trace x ===# 3

range : (n : Nat) -> Literal [n] Nat
range n = cast (Vect.range n)

product1 : (x : Nat) -> product (the (List Nat) [x]) = x
product1 x = rewrite plusZeroRightNeutral x in Refl

iidKolmogorovSmirnov :
  {shape : _} -> Tensor shape F64 -> (Tensor shape F64 -> Tensor shape F64) -> Tensor [] F64
iidKolmogorovSmirnov samples cdf =
  let n : Nat
      n = product shape

      indices : Tensor [n] F64 := cast (fromLiteral {dtype=U64} (range n))
      sampleSize : Tensor [] F64 := cast (fromLiteral {dtype=U64} (Scalar n))
      samplesFlat := reshape {sizesEqual=sym (product1 n)} {to=[n]} (cdf samples)
      deviationFromCDF : Tensor [n] F64 := indices / sampleSize - (sort (<) 0 samplesFlat)
   in reduce @{Max} [0] (abs deviationFromCDF)

covering
uniform : Property
uniform = withTests 20 . property $ do
  bound <- forAll (literal [5] finiteDoubles)
  bound' <- forAll (literal [5] finiteDoubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral bound
      bound' = fromLiteral bound'
      key = fromLiteral key
      seed = fromLiteral seed
      samples = evalState seed (uniform key (broadcast bound) (broadcast bound'))

      uniformCdf : Tensor [2000, 5] F64 -> Tensor [2000, 5] F64
      uniformCdf x = (x - broadcast bound) / broadcast (bound' - bound)

      ksTest := iidKolmogorovSmirnov samples uniformCdf

  diff (toLiteral ksTest) (<) 0.01

covering
uniformForNonFiniteBounds : Property
uniformForNonFiniteBounds = property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral [0.0, 0.0, 0.0, -inf, -inf, -inf, inf, inf, nan]
      bound' = fromLiteral [-inf, inf, nan, -inf, inf, nan, inf, nan, nan]
      key = fromLiteral key
      seed = fromLiteral seed
      samples = evalState seed (uniform key (broadcast bound) (broadcast bound'))

  samples ===# fromLiteral [-inf, inf, nan, -inf, nan, nan, inf, nan, nan]

covering
uniformForFiniteEqualBounds : Property
uniformForFiniteEqualBounds = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral [-1.0, 0.0, 1.0]
      key = fromLiteral key
      seed = fromLiteral seed
      samples = evalState seed (uniform key bound bound)

  samples ===# fromLiteral [-1.0, 0.0, 1.0]

covering
uniformSeedIsUpdated : Property
uniformSeedIsUpdated = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral bound
      bound' = fromLiteral bound'
      key = fromLiteral key
      seed = fromLiteral seed

      rng = uniform key {shape=[10]} (broadcast bound) (broadcast bound')
      (seed', sample) = runState seed rng
      (seed'', sample') = runState seed' rng

  diff (toLiteral seed') (/=) (toLiteral seed)
  diff (toLiteral seed'') (/=) (toLiteral seed')
  diff (toLiteral sample') (/=) (toLiteral sample)

covering
uniformIsReproducible : Property
uniformIsReproducible = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral bound
      bound' = fromLiteral bound'
      key = fromLiteral key
      seed = fromLiteral seed

      rng = uniform {shape=[10]} key (broadcast bound) (broadcast bound')
      sample = evalState seed rng
      sample' = evalState seed rng

  sample ===# sample'

covering
normal : Property
normal = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let key = fromLiteral key
      seed = fromLiteral seed

      samples : Tensor [100, 100] F64 = evalState seed (normal key)

      normalCdf : {shape : _} -> Tensor shape F64 -> Tensor shape F64
      normalCdf x = (fill 1.0 + erf (x / sqrt (fill 2.0))) / fill 2.0

      ksTest := iidKolmogorovSmirnov samples normalCdf

  diff (toLiteral ksTest) (<) 0.017

covering
normalSeedIsUpdated : Property
normalSeedIsUpdated = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let key = fromLiteral key
      seed = fromLiteral seed
      rng = normal key {shape=[10]}
      (seed', sample) = runState seed rng
      (seed'', sample') = runState seed' rng

  diff (toLiteral seed') (/=) (toLiteral seed)
  diff (toLiteral seed'') (/=) (toLiteral seed')
  diff (toLiteral sample') (/=) (toLiteral sample)

covering
normalIsReproducible : Property
normalIsReproducible = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let key = fromLiteral key
      seed = fromLiteral seed

      rng = normal {shape=[10]} key
      sample = evalState seed rng
      sample' = evalState seed rng

  sample ===# sample'

export covering
group : Group
group = MkGroup "Tensor" $ [
      ("toLiteral . fromLiteral", fromLiteralThenToLiteral)
    , ("can read/write finite numeric bounds to/from XLA", canConvertAtXlaNumericBounds)
    , ("show", show)
    , ("cast", cast)
    , ("reshape", reshape)
    , ("MultiSlice.slice", MultiSlice.slice)
    , ("slice", TestTensor.slice)
    , ("slice for variable index", sliceForVariableIndex)
    , ("concat", concat)
    , ("diag", diag)
    , ("triangle", triangle)
    , ("identity", identity)
    , ("expand", expand)
    , ("broadcast", broadcast)
    , ("squeeze", squeeze)
    , ("(.T)", (.T))
    , ("transpose", transpose)
    , ("map", mapResult)
    , ("map with non-trivial function", mapNonTrivial)
    , ("map2", map2Result)
    , ("map2 with re-used function arguments", map2ResultWithReusedFnArgs)
    , ("reduce", reduce)
    , ("sort", sort)
    , ("sort with empty axis", sortWithEmptyAxis)
    , ("sort with repeated elements", sortWithRepeatedElements)
    , ("reverse", reverse)
    , ("Vector.(@@)", Vector.(@@))
    , ("Matrix.(@@)", Matrix.(@@))
  ]
  ++ testElementwiseComparatorCases
  ++ testElementwiseUnaryCases
  ++ testElementwiseBinaryCases
  ++ [
      ("Scalarwise.(*)", scalarMultiplication)
    , ("Scalarwise.(/)", scalarDivision)
    , ("Sum", neutralIsNeutralForSum)
    , ("Prod", neutralIsNeutralForProd)
    , ("min F64", minF64)
    , ("max F64", maxF64)
    , ("Min", neutralIsNeutralForMin)
    , ("Max", neutralIsNeutralForMax)
    , ("Any", neutralIsNeutralForAny)
    , ("All", neutralIsNeutralForAll)
    , ("select", select)
    , ("cond for trivial usage", condResultTrivialUsage)
    , ("cond for re-used arguments", condResultWithReusedArgs)
    , ("erf", erf)
    , ("cholesky", cholesky)
    , (#"(|\) and (/|) result and inverse"#, triangularSolveResultAndInverse)
    , (#"(|\) and (/|) ignore opposite elements"#, triangularSolveIgnoresOppositeElems)
    , ("trace", trace)
    , ("uniform", uniform)
    , ("uniform for infinite and NaN bounds", uniformForNonFiniteBounds)
    , ("uniform is not NaN for finite equal bounds", uniformForFiniteEqualBounds)
    , ("uniform updates seed", uniformSeedIsUpdated)
    , ("uniform produces same samples for same seed", uniformIsReproducible)
    , ("normal", normal)
    , ("normal updates seed", normalSeedIsUpdated)
    , ("normal produces same samples for same seed", normalIsReproducible)
  ]
