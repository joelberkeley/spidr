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

partial
fromLiteralThenToLiteral : Property
fromLiteralThenToLiteral = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  x ==~ toLiteral (fromLiteral {dtype=F64} x)

  x <- forAll (literal shape int32s)
  x === toLiteral (fromLiteral {dtype=S32} x)

  x <- forAll (literal shape nats)
  x === toLiteral (fromLiteral {dtype=U32} x)

  x <- forAll (literal shape nats)
  x === toLiteral (fromLiteral {dtype=U64} x)

  x <- forAll (literal shape bool)
  x === toLiteral (fromLiteral {dtype=PRED} x)

[Finite] Bounded (Literal [] Double) where
  min = Scalar (min @{Finite})
  max = Scalar (max @{Finite})

partial
canConvertAtXlaNumericBounds : Property
canConvertAtXlaNumericBounds = fixedProperty $ do
  let f64min : Literal [] Double = min @{Finite}
      f64max : Literal [] Double = max @{Finite}
      min' : Shared $ Tensor [] F64 = Types.min @{Finite}
      max' : Shared $ Tensor [] F64 = Types.max @{Finite}
  toLiteral min' === f64min
  toLiteral max' === f64max
  toLiteral (do !(fromLiteral f64min) == !min') === True
  toLiteral (do !(fromLiteral f64max) == !max') === True

  let s32min : Literal [] Int32 = Scalar min
      s32max : Literal [] Int32 = Scalar max
      min' : Shared $ Tensor [] S32 = Types.min @{Finite}
      max' : Shared $ Tensor [] S32 = Types.max @{Finite}
  toLiteral min' === s32min
  toLiteral max' === s32max
  toLiteral (do !(fromLiteral s32min) == !min') === True
  toLiteral (do !(fromLiteral s32max) == !max') === True

  let u32min : Literal [] Nat = 0
      u32max : Literal [] Nat = 4294967295
      min' : Shared $ Tensor [] U32 = Types.min @{Finite}
      max' : Shared $ Tensor [] U32 = Types.max @{Finite}
  toLiteral min' === u32min
  toLiteral max' === u32max
  toLiteral (do !(fromLiteral u32min) == !min') === True
  toLiteral (do !(fromLiteral u32max) == !max') === True

  let u64min : Literal [] Nat = 0
      u64max : Literal [] Nat = 18446744073709551615
      min' : Shared $ Tensor [] U64 = Types.min @{Finite}
      max' : Shared $ Tensor [] U64 = Types.max @{Finite}
  toLiteral min' === u64min
  toLiteral max' === u64max
  toLiteral (do !(fromLiteral u64min) == !min') === True
  toLiteral (do !(fromLiteral u64max) == !max') === True

partial
boundedNonFinite : Property
boundedNonFinite = fixedProperty $ do
  let min' : Shared $ Tensor [] S32 = Types.min @{NonFinite}
      max' : Shared $ Tensor [] S32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Shared $ Tensor [] U32 = Types.min @{NonFinite}
      max' : Shared $ Tensor [] U32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Shared $ Tensor [] U64 = Types.min @{NonFinite}
      max' : Shared $ Tensor [] U64 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  Types.min @{NonFinite} ===# fromLiteral (-inf)
  Types.max @{NonFinite} ===# fromLiteral inf
  toLiteral {dtype=F64} (Types.min @{NonFinite}) === -inf
  toLiteral {dtype=F64} (Types.max @{NonFinite}) === inf

partial
show : Property
show = fixedProperty $ do
  let x : Shared $ Tensor [] S32 = 1
  show x === "constant, shape=[], metadata={:0}"

  let x : Shared $ Tensor [] S32 = (do !1 + !2)
  show x ===
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = fromLiteral {dtype=F64} [1.3, 2.0, -0.4]
  show x === "constant, shape=[3], metadata={:0}"

partial
cast : Property
cast = property $ do
  shape <- forAll shapes

  lit <- forAll (literal shape nats)
  let x : Shared $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=U32} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape nats)
  let x : Shared $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=U64} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape int32s)
  let x : Shared $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=S32} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

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
slice : Property
slice = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  (do slice [at 0] !x) ===# fromLiteral 3
  (do slice [at 1] !x) ===# fromLiteral 4
  (do slice [at 2] !x) ===# fromLiteral 5
  (do slice [0.to 0] !x) ===# fromLiteral []
  (do slice [0.to 1] !x) ===# fromLiteral [3]
  (do slice [0.to 2] !x) ===# fromLiteral [3, 4]
  (do slice [0.to 3] !x) ===# fromLiteral [3, 4, 5]
  (do slice [1.to 1] !x) ===# fromLiteral []
  (do slice [1.to 2] !x) ===# fromLiteral [4]
  (do slice [1.to 3] !x) ===# fromLiteral [4, 5]
  (do slice [2.to 2] !x) ===# fromLiteral []
  (do slice [2.to 3] !x) ===# fromLiteral [5]
{-
  (do slice [at (!(fromLiteral 0))] !x) ===# fromLiteral 3
  (do slice [at (!(fromLiteral 1))] !x) ===# fromLiteral 4
  (do slice [at (!(fromLiteral 2))] !x) ===# fromLiteral 5
  (do slice [at (!(fromLiteral 3))] !x) ===# fromLiteral 5
  (do slice [at (!(fromLiteral 5))] !x) ===# fromLiteral 5
  (do slice [(!(fromLiteral 0)).size 0] !x) ===# fromLiteral []
  (do slice [(!(fromLiteral 0)).size 1] !x) ===# fromLiteral [3]
  (do slice [(!(fromLiteral 0)).size 2] !x) ===# fromLiteral [3, 4]
  (do slice [(!(fromLiteral 0)).size 3] !x) ===# fromLiteral [3, 4, 5]
  (do slice [(!(fromLiteral 1)).size 0] !x) ===# fromLiteral []
  (do slice [(!(fromLiteral 1)).size 1] !x) ===# fromLiteral [4]
  (do slice [(!(fromLiteral 1)).size 2] !x) ===# fromLiteral [4, 5]
  (do slice [(!(fromLiteral 1)).size 3] !x) ===# fromLiteral [3, 4, 5]
  (do slice [(!(fromLiteral 2)).size 0] !x) ===# fromLiteral []
  (do slice [(!(fromLiteral 2)).size 1] !x) ===# fromLiteral [5]
  (do slice [(!(fromLiteral 3)).size 0] !x) ===# fromLiteral []
  (do slice [(!(fromLiteral 3)).size 1] !x) ===# fromLiteral [5]
  (do slice [(!(fromLiteral 3)).size 3] !x) ===# fromLiteral [3, 4, 5]
  (do slice [(!(fromLiteral 5)).size 0] !x) ===# fromLiteral []
  (do slice [(!(fromLiteral 5)).size 1] !x) ===# fromLiteral [5]
  (do slice [(!(fromLiteral 5)).size 3] !x) ===# fromLiteral [3, 4, 5]
      -}

  let idx : Nat
      idx = 2

  (do slice [at idx] !x) ===# fromLiteral 5

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  (do slice [0.to 1] !x) ===# fromLiteral [[3, 4, 5]]
  (do slice [1.to 1] !x) ===# fromLiteral []
  (do slice [all, 2.to 2] !x) ===# fromLiteral [[], []]
  (do slice [all, 1.to 3] !x) ===# fromLiteral [[4, 5], [7, 8]]
  (do slice [at 0, 2.to 2] !x) ===# fromLiteral []
  (do slice [at 0, 1.to 3] !x) ===# fromLiteral [4, 5]
  (do slice [at 1, 2.to 2] !x) ===# fromLiteral []
  (do slice [at 1, 1.to 3] !x) ===# fromLiteral [7, 8]
  (do slice [0.to 1, at 0] !x) ===# fromLiteral [3]
  (do slice [0.to 1, at 1] !x) ===# fromLiteral [4]
  (do slice [0.to 1, at 2] !x) ===# fromLiteral [5]
  (do slice [1.to 2, at 0] !x) ===# fromLiteral [6]
  (do slice [1.to 2, at 1] !x) ===# fromLiteral [7]
  (do slice [1.to 2, at 2] !x) ===# fromLiteral [8]
{-
  (do idx <- fromLiteral 0
      slice [idx.size 1] !x) ===# fromLiteral [[3, 4, 5]]
  (do idx <- fromLiteral 1
      slice [idx.size 0] !x) ===# fromLiteral []
  (do idx <- fromLiteral 2
      slice [idx.size 0] !x) ===# fromLiteral []
  (do idx <- fromLiteral 2
      slice [idx.size 1] !x) ===# fromLiteral [[6, 7, 8]]
  (do idx <- fromLiteral 4
      slice [idx.size 0] !x) ===# fromLiteral []
  (do idx <- fromLiteral 4
      slice [idx.size 1] !x) ===# fromLiteral [[6, 7, 8]]
  (do idx <- fromLiteral 2
      slice [all, idx.size 0] !x) ===# fromLiteral [[], []]
  (do idx <- fromLiteral 1
      slice [all, idx.size 2] !x) ===# fromLiteral [[4, 5], [7, 8]]
  (do idx <- fromLiteral 3
      slice [all, idx.size 0] !x) ===# fromLiteral [[], []]
  (do idx <- fromLiteral 3
      slice [all, idx.size 2] !x) ===# fromLiteral [[4, 5], [7, 8]]
  (do idx <- fromLiteral 5
      slice [all, idx.size 0] !x) ===# fromLiteral [[], []]
  (do idx <- fromLiteral 5
      slice [all, idx.size 2] !x) ===# fromLiteral [[4, 5], [7, 8]]
  (do idx <- fromLiteral 2
      slice [at 0, idx.size 0] !x) ===# fromLiteral []
  (do slice [at 0, (!(fromLiteral 1)).size 2] !x) ===# fromLiteral [4, 5]
  (do slice [at 1, (!(fromLiteral 2)).size 0] !x) ===# fromLiteral []
  (do slice [at 1, (!(fromLiteral 1)).size 2] !x) ===# fromLiteral [7, 8]
  (do slice [at 1, (!(fromLiteral 3)).size 0] !x) ===# fromLiteral []
  (do slice [at 1, (!(fromLiteral 3)).size 2] !x) ===# fromLiteral [7, 8]
  (do slice [at 1, (!(fromLiteral 5)).size 0] !x) ===# fromLiteral []
  (do slice [at 1, (!(fromLiteral 5)).size 2] !x) ===# fromLiteral [7, 8]
  (do slice [(!(fromLiteral 0)).size 1, at 0] !x) ===# fromLiteral [3]
  (do slice [(!(fromLiteral 0)).size 1, at 1] !x) ===# fromLiteral [4]
  (do slice [(!(fromLiteral 0)).size 1, at 2] !x) ===# fromLiteral [5]
  (do slice [(!(fromLiteral 1)).size 1, at 0] !x) ===# fromLiteral [6]
  (do slice [(!(fromLiteral 1)).size 1, at 1] !x) ===# fromLiteral [7]
  (do slice [(!(fromLiteral 1)).size 1, at 2] !x) ===# fromLiteral [8]
  (do slice [(!(fromLiteral 2)).size 1, at 0] !x) ===# fromLiteral [6]
  (do slice [(!(fromLiteral 2)).size 1, at 1] !x) ===# fromLiteral [7]
  (do slice [(!(fromLiteral 2)).size 1, at 2] !x) ===# fromLiteral [8]
  (do slice [(!(fromLiteral 4)).size 1, at 0] !x) ===# fromLiteral [6]
  (do slice [(!(fromLiteral 4)).size 1, at 1] !x) ===# fromLiteral [7]
  (do slice [(!(fromLiteral 4)).size 1, at 2] !x) ===# fromLiteral [8]
      -}

  let x : Array [60] Int32 = fromList [0..59]
      x = (do reshape {to=[2, 5, 3, 2]} !(fromLiteral {shape=[60]} {dtype=S32} $ cast x))

  let idx = fromLiteral {dtype=U64} 0
      start = fromLiteral {dtype=U64} 1
  (do slice [at 1, 2.to 5, (!start).size 2, at !idx] !x) ===# fromLiteral [[44, 46], [50, 52], [56, 58]]

index : (idx : Nat) -> {auto 0 inDim : LT idx n} -> Literal [n] a -> Literal [] a
index {inDim = (LTESucc _)} 0 (y :: _) = y
index {inDim = (LTESucc _)} (S k) (_ :: xs) = index k xs

partial
sliceForVariableIndex : Property
sliceForVariableIndex = property $ do
  idx <- forAll dims
  rem <- forAll dims
  lit <- forAll (literal [idx + S rem] nats)
  let x = fromLiteral {dtype=U32} lit
  index @{inDim} idx lit === toLiteral (do slice [at @{inDim} idx] !x)

  where
  %hint
  inDim : {idx, rem : _} -> LTE (S idx) (idx + S rem)
  inDim {idx = 0} = LTESucc LTEZero
  inDim {idx = (S k)} = LTESucc inDim

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
  (do (!(fromLiteral {dtype=S32} [])).T) ===# fromLiteral []
  (do (!(fromLiteral {dtype=S32} [[3]])).T) ===# fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = fromLiteral [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  (do (!x).T) ===# expected

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
      x : Shared $ Tensor [2, 3, 4, 5] S32 = (do reshape !(fromLiteral {shape=[120]} (cast x)))
  (do transpose [0, 1, 2, 3] !x) ===# x
  (do slice [all, at 1, at 0] !(transpose [0, 2, 1, 3] !x)) ===# (do slice [all, at 0, at 1] !x)
  (do slice [at 2, at 4, at 0, at 1] !(transpose [2, 3, 1, 0] !x)) ===# (do slice [at 1, at 0, at 2, at 4] !x)

partial
mapResult : Property
mapResult = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral x
  map (1.0 /) x ==~ toLiteral (do map (!1.0 /) !x')

  x <- forAll (literal shape int32s)
  let x' = fromLiteral {dtype=S32} x
  map (+ 1) x === toLiteral (do map (+ !1) !x')

partial
mapNonTrivial : Property
mapNonTrivial = fixedProperty $ do
  (do map {a=S32} (\x => x + x) !1) ===# 2
  (do map {a=S32} (\_ => 2) !1) ===# 2
  (do map {a=S32} (map (+ !1)) !1) ===# 2

partial
map2Result : Property
map2Result = fixedProperty $ do
  shape <- forAll shapes

  let int32s = literal shape int32s
  [x, y] <- forAll (np [int32s, int32s])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x + y |] === toLiteral (do map2 Tensor.(+) !x' !y')

  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x + y |] ==~ toLiteral (do map2 Tensor.(+) !x' !y')

partial
map2ResultWithReusedFnArgs : Property
map2ResultWithReusedFnArgs = fixedProperty $ do
  let x : Shared (Tensor [] S32) = 6
  (do map2 (\x, y => !(!(x + x) + y) + y) !1 !2) ===# x 

partial
reduce : Property
reduce = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [1] !x) ===# fromLiteral [6, -6]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-2, -3, -4]]
  (do reduce @{Sum} [0, 1] !x) ===# fromLiteral (-3)

  let x = fromLiteral {dtype=S32} [[[1], [2], [3]], [[-2], [-3], [-4]]]
  (do reduce @{Sum} [0, 1] !x) ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3]], [[-2, -3, -4]]]
  (do reduce @{Sum} [0, 2] !x) ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3], [-2, -3, -4]]]
  (do reduce @{Sum} [1, 2] !x) ===# fromLiteral [-3]

  let x = fromLiteral {dtype=S32} [[[1, 2, 3], [4, 5, 6]], [[-2, -3, -4], [-6, -7, -8]]]
  (do reduce @{Sum} [0, 2] !x) ===# fromLiteral [-3, -6]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
  (do reduce @{Sum} [0] !x) ===# fromLiteral [0, 0, 0]

  let x = fromLiteral {dtype=PRED} [[True, False, True], [True, False, False]]
  (do reduce @{All} [1] !x) ===# fromLiteral [False, False]

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

partial
sort : Property
sort = withTests 20 . property $ do
  d <- forAll dims
  dd <- forAll dims
  ddd <- forAll dims

  x <- forAll (literal [S d] int32s)
  let x = fromLiteral {dtype=S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  x <- forAll (literal [S d, S dd] int32s)
  let x = fromLiteral {dtype=S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = (do sort (<) 1 !x)
      init = (do slice [all, 0.to dd] !sorted)
      tail = (do slice [all, 1.to (S dd)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  x <- forAll (literal [S d, S dd, S ddd] int32s)
  let x = fromLiteral {dtype=S32} x

  let sorted = (do sort (<) 0 !x)
      init = (do slice [0.to d] !sorted)
      tail = (do slice [1.to (S d)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = (do sort (<) 1 !x)
      init = (do slice [all, 0.to dd] !sorted)
      tail = (do slice [all, 1.to (S dd)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

  let sorted = (do sort (<) 2 !x)
      init = (do slice [all, all, 0.to ddd] !sorted)
      tail = (do slice [all, all, 1.to (S ddd)] !sorted)
  diff (toLiteral init) (\x, y => all [| x <= y |]) (toLiteral tail)

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
  let x = fromLiteral {shape=[0, 2, 3]} {dtype=S32} []
  (do sort (<) 0 !x) ===# x

  let x = fromLiteral {shape=[0, 2, 3]} {dtype=S32} []
  (do sort (<) 1 !x) ===# x

  let x = fromLiteral {shape=[2, 0, 3]} {dtype=S32} [[], []]
  (do sort (<) 0 !x) ===# x

  let x = fromLiteral {shape=[2, 0, 3]} {dtype=S32} [[], []]
  (do sort (<) 1 !x) ===# x

partial
sortWithRepeatedElements : Property
sortWithRepeatedElements = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [1, 3, 4, 3, 2]
  (do sort (<) 0 !x) ===# fromLiteral [1, 2, 3, 3, 4]

  let x = fromLiteral {dtype=S32} [[1, 4, 4], [3, 2, 5]]
  (do sort (<) 0 !x) ===# fromLiteral [[1, 2, 4], [3, 4, 5]]
  (do sort (<) 1 !x) ===# fromLiteral [[1, 4, 4], [2, 3, 5]]

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

namespace Vector
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [-2, 0, 1]
        r = fromLiteral {dtype=S32} [3, 1, 2]
    (do !l @@ !r) ===# -4

namespace Matrix
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [3, 3, -1]
    (do !l @@ !r) ===# fromLiteral [-7, 8]

    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    (do !l @@ !r) ===# fromLiteral [[ -7,  -2], [  8, -11]]

namespace S32
  export partial
  testElementwiseUnary :
    (Int32 -> Int32) ->
    (forall shape . Tensor shape S32 -> Shared $ Tensor shape S32) ->
    Property
  testElementwiseUnary fInt fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape int32s)
    let x' = fromLiteral x
    [| fInt x |] === toLiteral (do fTensor !x')

namespace F64
  export partial
  testElementwiseUnary :
    (Double -> Double) ->
    (forall shape . Tensor shape F64 -> Shared $ Tensor shape F64) ->
    Property
  testElementwiseUnary fDouble fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape doubles)
    let x' = fromLiteral x
    [| fDouble x |] ==~ toLiteral (do fTensor !x')

namespace PRED
  export partial
  testElementwiseUnary :
    (Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Shared $ Tensor shape PRED) ->
    Property
  testElementwiseUnary fBool fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape bool)
    let x' = fromLiteral x
    [| fBool x |] === toLiteral (do fTensor !x')

partial
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
  export partial
  testElementwiseBinary :
    (Int32 -> Int32 -> Int32) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Shared $ Tensor shape S32) ->
    Property
  testElementwiseBinary fInt fTensor = property $ do
    shape <- forAll shapes
    let int32s = literal shape int32s
    [x, y] <- forAll (np [int32s, int32s])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (do fTensor !x' !y')

namespace F64
  export partial
  testElementwiseBinary :
    (Double -> Double -> Double) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Shared $ Tensor shape F64) ->
    Property
  testElementwiseBinary fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] ==~ toLiteral (do fTensor !x' !y')

namespace PRED
  export partial
  testElementwiseBinary :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Shared $ Tensor shape PRED) ->
    Property
  testElementwiseBinary fBool fTensor = property $ do
    shape <- forAll shapes
    let bools = literal shape bool
    [x, y] <- forAll (np [bools, bools])
    let x' = fromLiteral {dtype=PRED} x
        y' = fromLiteral {dtype=PRED} y
    [| fBool x y |] === toLiteral (do fTensor !x' !y')

partial
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
    ("min F64", F64.testElementwiseBinary min' min),
    ("max F64", F64.testElementwiseBinary max' max),
    ("(&&)", PRED.testElementwiseBinary and (&&)),
    ("(||)", PRED.testElementwiseBinary or (||))
  ]

  where
  min' : Double -> Double -> Double
  min' x y = if x == x && y == y then min x y else nan

  max' : Double -> Double -> Double
  max' x y = if x == x && y == y then max x y else nan

  and : Bool -> Bool -> Bool
  and x y = x && y

  or : Bool -> Bool -> Bool
  or x y = x || y

partial
scalarMultiplication : Property
scalarMultiplication = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (scalar *) lit ==~ toLiteral (do !scalar' * !lit')

partial
scalarDivision : Property
scalarDivision = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (/ scalar) lit ==~ toLiteral (do !lit' / !scalar')

partial
neutralIsNeutralForSum : Property
neutralIsNeutralForSum = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (do (<+>) @{Sum} !x' !(neutral @{Sum}))
      left = (do (<+>) @{Sum} !(neutral @{Sum}) !x')
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape int32s)
  let x' = fromLiteral {dtype=S32} x
      right = (do (<+>) @{Sum} !x' !(neutral @{Sum}))
      left = (do (<+>) @{Sum} !(neutral @{Sum}) !x')
  toLiteral right === x
  toLiteral left === x

partial
neutralIsNeutralForProd : Property
neutralIsNeutralForProd = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (do (<+>) @{Prod} !x' !(neutral @{Prod}))
      left = (do (<+>) @{Prod} !(neutral @{Prod}) !x')
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape int32s)
  let x' = fromLiteral {dtype=S32} x
      right = (do (<+>) @{Prod} !x' !(neutral @{Prod}))
      left = (do (<+>) @{Prod} !(neutral @{Prod}) !x')
  toLiteral right === x
  toLiteral left === x

partial
neutralIsNeutralForAny : Property
neutralIsNeutralForAny = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (do (<+>) @{Any} !x' !(neutral @{MonoidM.Any}))
      left = (do (<+>) @{Any} !(neutral @{MonoidM.Any}) !x')
  toLiteral right === x
  toLiteral left === x

partial
neutralIsNeutralForAll : Property
neutralIsNeutralForAll = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (do (<+>) @{All} !x' !(neutral @{MonoidM.All}))
      left = (do (<+>) @{All} !(neutral @{MonoidM.All}) !x')
  toLiteral right === x
  toLiteral left === x

partial
neutralIsNeutralForMin : Property
neutralIsNeutralForMin = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (do (<+>) @{Min} !x' !(neutral @{Min}))
      left = (do (<+>) @{Min} !(neutral @{Min}) !x')
  toLiteral right ==~ x
  toLiteral left ==~ x

partial
neutralIsNeutralForMax : Property
neutralIsNeutralForMax = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (do (<+>) @{Max} !x' !(neutral @{Max}))
      left = (do (<+>) @{Max} !(neutral @{Max}) !x')
  toLiteral right ==~ x
  toLiteral left ==~ x

partial
argmin : Property
argmin = property $ do
  d <- forAll dims
  xs <- forAll (literal [S d] doubles)
  let xs = fromLiteral xs
  (do slice [at !(argmin !xs)] !xs) ===# (do reduce [0] @{Min} !xs)

partial
argmax : Property
argmax = property $ do
  d <- forAll dims
  xs <- forAll (literal [S d] doubles)
  let xs = fromLiteral xs
  (do slice [at !(argmax !xs)] !xs) ===# (do reduce [0] @{Max} !xs)

namespace S32
  export partial
  testElementwiseComparator :
    (Int32 -> Int32 -> Bool) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Shared $ Tensor shape PRED) ->
    Property
  testElementwiseComparator fInt fTensor = property $ do
    shape <- forAll shapes
    let int32s = literal shape int32s
    [x, y] <- forAll (np [int32s, int32s])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (do fTensor !x' !y')

namespace F64
  export partial
  testElementwiseComparator :
    (Double -> Double -> Bool) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Shared $ Tensor shape PRED) ->
    Property
  testElementwiseComparator fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] === toLiteral (do fTensor !x' !y')

namespace PRED
  export partial
  testElementwiseComparator :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Shared $ Tensor shape PRED) ->
    Property
  testElementwiseComparator = testElementwiseBinary

partial
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

partial
select : Property
select = fixedProperty $ do
  let onTrue = fromLiteral {dtype=S32} 1
      onFalse = fromLiteral 0
  (do select !(fromLiteral True) !onTrue !onFalse) ===# onTrue
  (do select !(fromLiteral False) !onTrue !onFalse) ===# onFalse

  let pred = fromLiteral [[False, True, True], [True, False, False]]
      onTrue = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
      onFalse = fromLiteral [[6, 7, 8], [9, 10, 11]]
      expected = fromLiteral [[6, 1, 2], [3, 10, 11]]
  (do select !pred !onTrue !onFalse) ===# expected
{-
partial
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

partial
condResultWithReusedArgs : Property
condResultWithReusedArgs = fixedProperty $ do
  let x = fromLiteral {dtype=S32} 1
      y = fromLiteral {dtype=S32} 3
  cond (fromLiteral True) (\z => z + z) x (\z => z * z) y ===# 2
  cond (fromLiteral False) (\z => z + z) x (\z => z * z) y ===# 9
  -}
partial
erf : Property
erf = fixedProperty $ do
  let x = fromLiteral [-1.5, -0.5, 0.5, 1.5]
      expected = fromLiteral [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  (do erf !x) ===# expected

partial
cholesky : Property
cholesky = fixedProperty $ do
  let x = fromLiteral [[1.0, 0.0], [2.0, 0.0]]
      expected = fromLiteral [[nan, 0], [nan, nan]]
  (do cholesky !x) ===# expected

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
  (do cholesky !x) ===# expected

partial
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
      actual = (do !a |\ !b)
      expected = fromLiteral [
                    [ 0.52820396,  0.43452972],
                    [ 0.79913783, -0.10254406],
                    [ 1.8257918 ,  2.7147462 ]
                  ]
  actual ===# expected
  (do !a @@ !actual) ===# b

  let actual = (do !((!a).T) \| !b)
      expected = fromLiteral [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  actual ===# expected
  (do !((!a).T) @@ !actual) ===# b

partial
triangularSolveIgnoresOppositeElems : Property
triangularSolveIgnoresOppositeElems = fixedProperty $ do
  let a = fromLiteral [[1.0, 2.0], [3.0, 4.0]]
      aLower = fromLiteral [[1.0, 0.0], [3.0, 4.0]]
      b = fromLiteral [5.0, 6.0]
  (do !a |\ !b) ===# (do !aLower |\ !b)

  let aUpper = fromLiteral [[1.0, 2.0], [0.0, 4.0]]
  (do !a \| !b) ===# (do !aUpper \| !b)

partial
trace : Property
trace = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[-1, 5], [1, 4]]
  (do trace !x) ===# 3
{-
range : (n : Nat) -> Literal [n] Nat
range n = cast (Vect.range n)

product1 : (x : Nat) -> product (the (List Nat) [x]) = x
product1 x = rewrite plusZeroRightNeutral x in Refl

partial
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

partial
uniform : Property
uniform = withTests 20 . property $ do
  bound <- forAll (literal [5] finiteDoubles)
  bound' <- forAll (literal [5] finiteDoubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral bound
      bound' = fromLiteral bound'
      bound' = select (bound' == bound) (bound' + fill 1.0e-9) bound'
      key = fromLiteral key
      seed = fromLiteral seed
      samples = evalState seed (uniform key (broadcast bound) (broadcast bound'))

      uniformCdf : Tensor [2000, 5] F64 -> Tensor [2000, 5] F64
      uniformCdf x = (x - broadcast bound) / broadcast (bound' - bound)

      ksTest := iidKolmogorovSmirnov samples uniformCdf

  diff (toLiteral ksTest) (<) 0.015

partial
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

partial
uniformForFiniteEqualBounds : Property
uniformForFiniteEqualBounds = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral [min @{Finite}, -1.0, -1.0e-308, 0.0, 1.0e-308, 1.0, max @{Finite}]
      key = fromLiteral key
      seed = fromLiteral seed
      samples = evalState seed (uniform key bound bound)

  samples ===# bound

partial
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

partial
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

partial
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

  diff (toLiteral ksTest) (<) 0.02

partial
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

partial
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
-}

export partial
group : Group
group = MkGroup "Tensor" $ [
      ("toLiteral . fromLiteral", fromLiteralThenToLiteral)
    , ("can read/write finite numeric bounds to/from XLA", canConvertAtXlaNumericBounds)
    , ("bounded non-finite", boundedNonFinite)
    , ("show", show)
    , ("cast", cast)
    , ("reshape", reshape)
    , ("MultiSlice.slice", MultiSlice.slice)
--    , ("slice", TestTensor.slice)
--    , ("slice for variable index", sliceForVariableIndex)
    , ("concat", concat)
    , ("diag", diag)
    , ("triangle", triangle)
    , ("identity", identity)
    , ("expand", expand)
    , ("broadcast", broadcast)
    , ("squeeze", squeeze)
    , ("(.T)", (.T))
--    , ("transpose", transpose) -- test uses slice
    , ("map", mapResult)
    , ("map with non-trivial function", mapNonTrivial)
    , ("map2", map2Result)
    , ("map2 with re-used function arguments", map2ResultWithReusedFnArgs)
    , ("reduce", reduce)
--    , ("sort", sort) -- test uses slice
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
--    , ("argmin", argmin) -- test uses slice
--    , ("argmax", argmax) -- test uses slice
    , ("Min", neutralIsNeutralForMin)
    , ("Max", neutralIsNeutralForMax)
    , ("Any", neutralIsNeutralForAny)
    , ("All", neutralIsNeutralForAll)
    , ("select", select)
--    , ("cond for trivial usage", condResultTrivialUsage)
--    , ("cond for re-used arguments", condResultWithReusedArgs)
    , ("erf", erf)
    , ("cholesky", cholesky)
    , (#"(|\) and (/|) result and inverse"#, triangularSolveResultAndInverse)
    , (#"(|\) and (/|) ignore opposite elements"#, triangularSolveIgnoresOppositeElems)
    , ("trace", trace)
--    , ("uniform", uniform)
--    , ("uniform for infinite and NaN bounds", uniformForNonFiniteBounds)
--    , ("uniform is not NaN for finite equal bounds", uniformForFiniteEqualBounds)
--    , ("uniform updates seed", uniformSeedIsUpdated)
--    , ("uniform produces same samples for same seed", uniformIsReproducible)
--    , ("normal", normal)
--    , ("normal updates seed", normalSeedIsUpdated)
--    , ("normal produces same samples for same seed", normalIsReproducible)
  ]
