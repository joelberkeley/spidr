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

import Unit.TestTensor.Elementwise
import Unit.TestTensor.HigherOrder
import Unit.TestTensor.Sampling
import Unit.TestTensor.Slice
import Unit.TestTensor.Structure

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
  x ==~ unsafePerformIO (toLiteral (fromLiteral {dtype=F64} x))

  x <- forAll (literal shape int32s)
  x === unsafePerformIO (toLiteral (fromLiteral {dtype=S32} x))

  x <- forAll (literal shape nats)
  x === unsafePerformIO (toLiteral (fromLiteral {dtype=U32} x))

  x <- forAll (literal shape nats)
  x === unsafePerformIO (toLiteral (fromLiteral {dtype=U64} x))

  x <- forAll (literal shape bool)
  x === unsafePerformIO (toLiteral (fromLiteral {dtype=PRED} x))

partial
canConvertAtXlaNumericBounds : Property
canConvertAtXlaNumericBounds = fixedProperty $ do
  let f64min : Literal [] Double = min @{Finite}
      f64max : Literal [] Double = max @{Finite}
      min' : Ref $ Tensor [] F64 = Types.min @{Finite}
      max' : Ref $ Tensor [] F64 = Types.max @{Finite}
  unsafeToLiteral min' === f64min
  unsafeToLiteral max' === f64max
  unsafeToLiteral (fromLiteral f64min == min') === True
  unsafeToLiteral (fromLiteral f64max == max') === True

  let s32min : Literal [] Int32 = Scalar min
      s32max : Literal [] Int32 = Scalar max
      min' : Ref $ Tensor [] S32 = Types.min @{Finite}
      max' : Ref $ Tensor [] S32 = Types.max @{Finite}
  unsafeToLiteral min' === s32min
  unsafeToLiteral max' === s32max
  unsafeToLiteral (fromLiteral s32min == min') === True
  unsafeToLiteral (fromLiteral s32max == max') === True

  let u32min : Literal [] Nat = 0
      u32max : Literal [] Nat = 4294967295
      min' : Ref $ Tensor [] U32 = Types.min @{Finite}
      max' : Ref $ Tensor [] U32 = Types.max @{Finite}
  unsafeToLiteral min' === u32min
  unsafeToLiteral max' === u32max
  unsafeToLiteral (fromLiteral u32min == min') === True
  unsafeToLiteral (fromLiteral u32max == max') === True

  let u64min : Literal [] Nat = 0
      u64max : Literal [] Nat = 18446744073709551615
      min' : Ref $ Tensor [] U64 = Types.min @{Finite}
      max' : Ref $ Tensor [] U64 = Types.max @{Finite}
  unsafeToLiteral min' === u64min
  unsafeToLiteral max' === u64max
  unsafeToLiteral (fromLiteral u64min == min') === True
  unsafeToLiteral (fromLiteral u64max == max') === True

partial
boundedNonFinite : Property
boundedNonFinite = fixedProperty $ do
  let min' : Ref $ Tensor [] S32 = Types.min @{NonFinite}
      max' : Ref $ Tensor [] S32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Ref $ Tensor [] U32 = Types.min @{NonFinite}
      max' : Ref $ Tensor [] U32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Ref $ Tensor [] U64 = Types.min @{NonFinite}
      max' : Ref $ Tensor [] U64 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  Types.min @{NonFinite} ===# fromLiteral (-inf)
  Types.max @{NonFinite} ===# fromLiteral inf
  unsafeToLiteral {dtype=F64} (Types.min @{NonFinite}) === -inf
  unsafeToLiteral {dtype=F64} (Types.max @{NonFinite}) === inf

partial
show : Property
show = fixedProperty $ do
  let x : Ref $ Tensor [] S32 = 1
  show x === "constant, shape=[], metadata={:0}"

  let x : Ref $ Tensor [] S32 = 1 + 2
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
  let x : Ref $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=U32} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape nats)
  let x : Ref $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=U64} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

  lit <- forAll (literal shape int32s)
  let x : Ref $ Tensor shape F64 = (do castDtype !(fromLiteral {dtype=S32} lit))
  x ===# fromLiteral (map (cast {to=Double}) lit)

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

namespace Vector
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [-2, 0, 1]
        r = fromLiteral {dtype=S32} [3, 1, 2]
    l @@ r ===# -4

namespace Matrix
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [3, 3, -1]
    l @@ r ===# fromLiteral [-7, 8]

    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    l @@ r ===# fromLiteral [[ -7,  -2], [  8, -11]]

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

partial
triangularSolveIgnoresOppositeElems : Property
triangularSolveIgnoresOppositeElems = fixedProperty $ do
  let a = fromLiteral [[1.0, 2.0], [3.0, 4.0]]
      aLower = fromLiteral [[1.0, 0.0], [3.0, 4.0]]
      b = fromLiteral [5.0, 6.0]
  a |\ b ===# aLower |\ b

  let aUpper = fromLiteral [[1.0, 2.0], [0.0, 4.0]]
  a \| b ===# aUpper \| b

partial
trace : Property
trace = fixedProperty $ do
  let x = fromLiteral {dtype=S32} [[-1, 5], [1, 4]]
  (do trace !x) ===# 3

export partial
group : Group
group = MkGroup "Tensor" $ [
      ("toLiteral . fromLiteral", fromLiteralThenToLiteral)
    , ("can read/write finite numeric bounds to/from XLA", canConvertAtXlaNumericBounds)
    , ("bounded non-finite", boundedNonFinite)
    , ("show", show)
    , ("cast", cast)
    , ("identity", identity)
    , ("Vector.(@@)", Vector.(@@))
    , ("Matrix.(@@)", Matrix.(@@))
    , ("argmin", argmin)
    , ("argmax", argmax)
    , ("select", select)
    , ("erf", erf)
    , ("cholesky", cholesky)
    , (#"(|\) and (/|) result and inverse"#, triangularSolveResultAndInverse)
    , (#"(|\) and (/|) ignore opposite elements"#, triangularSolveIgnoresOppositeElems)
    , ("trace", trace)
  ] ++ concat (the (List _) [
      Unit.TestTensor.Elementwise.all
    , Unit.TestTensor.HigherOrder.all
    , Unit.TestTensor.Sampling.all
    , Unit.TestTensor.Slice.all
    , Unit.TestTensor.Structure.all
  ])
