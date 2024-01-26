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

import Data.Nat
import Data.Vect
import System

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases
import Utils.Proof

import Unit.TestTensor.Elementwise
import Unit.TestTensor.HigherOrder
import Unit.TestTensor.Sampling
import Unit.TestTensor.Slice
import Unit.TestTensor.Structure

partial
tensorThenEval : Property
tensorThenEval = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  x ==~ unsafePerformIO (eval (tensor {dtype=F64} x))

  x <- forAll (literal shape int32s)
  x === unsafePerformIO (eval (tensor {dtype=S32} x))

  x <- forAll (literal shape nats)
  x === unsafePerformIO (eval (tensor {dtype=U32} x))

  x <- forAll (literal shape nats)
  x === unsafePerformIO (eval (tensor {dtype=U64} x))

  x <- forAll (literal shape bool)
  x === unsafePerformIO (eval (tensor {dtype=PRED} x))

partial
evalTuple : Property
evalTuple = property $ do
  s0 <- forAll shapes
  s1 <- forAll shapes
  s2 <- forAll shapes

  x0 <- forAll (literal s0 doubles)
  x1 <- forAll (literal s1 int32s)
  x2 <- forAll (literal s2 nats)

  let y0 = tensor {dtype = F64} x0
      y1 = tensor {dtype = S32} x1
      y2 = tensor {dtype = U64} x2

  let [] = unsafePerformIO $ eval {tys = []} (pure [])

  let [x0'] = unsafePerformIO $ eval {tys = [_]} (do pure [!y0])

  x0' ==~ x0

  let [x0', x1'] = unsafePerformIO $ eval {tys = [_, _]} (do pure [!y0, !y1])

  x0' ==~ x0
  x1' === x1

  let [x0', x1', x2'] = unsafePerformIO $ eval {tys = [_, _, _]} (do pure [!y0, !y1, !y2])

  x0' ==~ x0
  x1' === x1
  x2' === x2

partial
evalTupleNonTrivial : Property
evalTupleNonTrivial = property $ do
  let xs = do y0 <- tensor [1.0, -2.0, 0.4]
              y1 <- tensor 3.0
              u <- exp y0
              v <- slice [at 1] u + pure y1
              w <- slice [0.to 2] u
              pure [v, w]

      [v, w] = unsafePerformIO $ eval {tys = [_, _]} xs

  v ==~ Scalar (exp (-2.0) + 3.0)
  w ==~ [| exp [1.0, -2.0] |]

partial
canConvertAtXlaNumericBounds : Property
canConvertAtXlaNumericBounds = fixedProperty $ do
  let f64min : Literal [] Double = min @{Finite}
      f64max : Literal [] Double = max @{Finite}
      min' : Graph $ Tensor [] F64 = Types.min @{Finite}
      max' : Graph $ Tensor [] F64 = Types.max @{Finite}
  unsafeEval min' === f64min
  unsafeEval max' === f64max
  unsafeEval (tensor f64min == min') === True
  unsafeEval (tensor f64max == max') === True

  let s32min : Literal [] Int32 = Scalar min
      s32max : Literal [] Int32 = Scalar max
      min' : Graph $ Tensor [] S32 = Types.min @{Finite}
      max' : Graph $ Tensor [] S32 = Types.max @{Finite}
  unsafeEval min' === s32min
  unsafeEval max' === s32max
  unsafeEval (tensor s32min == min') === True
  unsafeEval (tensor s32max == max') === True

  let u32min : Literal [] Nat = 0
      u32max : Literal [] Nat = 4294967295
      min' : Graph $ Tensor [] U32 = Types.min @{Finite}
      max' : Graph $ Tensor [] U32 = Types.max @{Finite}
  unsafeEval min' === u32min
  unsafeEval max' === u32max
  unsafeEval (tensor u32min == min') === True
  unsafeEval (tensor u32max == max') === True

  let u64min : Literal [] Nat = 0
      u64max : Literal [] Nat = 18446744073709551615
      min' : Graph $ Tensor [] U64 = Types.min @{Finite}
      max' : Graph $ Tensor [] U64 = Types.max @{Finite}
  unsafeEval min' === u64min
  unsafeEval max' === u64max
  unsafeEval (tensor u64min == min') === True
  unsafeEval (tensor u64max == max') === True

partial
boundedNonFinite : Property
boundedNonFinite = fixedProperty $ do
  let min' : Graph $ Tensor [] S32 = Types.min @{NonFinite}
      max' : Graph $ Tensor [] S32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Graph $ Tensor [] U32 = Types.min @{NonFinite}
      max' : Graph $ Tensor [] U32 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  let min' : Graph $ Tensor [] U64 = Types.min @{NonFinite}
      max' : Graph $ Tensor [] U64 = Types.max @{NonFinite}
  min' ===# Types.min @{Finite}
  max' ===# Types.max @{Finite}

  Types.min @{NonFinite} ===# tensor (-inf)
  Types.max @{NonFinite} ===# tensor inf
  unsafeEval {dtype=F64} (Types.min @{NonFinite}) === -inf
  unsafeEval {dtype=F64} (Types.max @{NonFinite}) === inf

partial
iota : Property
iota = property $ do
  init <- forAll shapes
  mid <- forAll dims
  tail <- forAll shapes

  let broadcastTail : Primitive dtype =>
                      {n : _} ->
                      (tail : Shape) ->
                      Tensor [n] dtype ->
                      Graph $ Tensor (n :: tail) dtype
      broadcastTail [] x = pure x
      broadcastTail (d :: ds) x = do
        x <- broadcastTail ds x
        broadcast !(expand 1 x)

  let rangeFull = do
        rangeV <- tensor {dtype = U64} $ cast (Vect.range mid)
        rangeVTail <- broadcastTail tail rangeV
        broadcast {shapesOK = broadcastableByLeading init} rangeVTail
      inBounds = appendNonEmptyLengthInBounds init mid tail
      actual : Graph (Tensor (init ++ mid :: tail) U64) = iota {inBounds} (length init)

  actual ===# rangeFull

  let actual : Graph (Tensor (init ++ mid :: tail) F64) = iota {inBounds} (length init)

  actual ===# (do castDtype !rangeFull)

partial
iotaExamples : Property
iotaExamples = fixedProperty $ do
  iota 0 ===# tensor {dtype = S32} [0, 1, 2, 3]
  iota 1 ===# tensor {dtype = S32} [[0], [0], [0], [0]]

  iota 1 ===# tensor {dtype = S32} [[0, 1, 2, 3, 4],
                                    [0, 1, 2, 3, 4],
                                    [0, 1, 2, 3, 4]]

  iota 0 ===# tensor {dtype = S32} [[0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1],
                                    [2, 2, 2, 2, 2]]

  iota 1 ===# tensor {dtype = F64} [[0.0, 1.0, 2.0, 3.0, 4.0],
                                    [0.0, 1.0, 2.0, 3.0, 4.0],
                                    [0.0, 1.0, 2.0, 3.0, 4.0]]

  iota 0 ===# tensor {dtype = F64} [[0.0, 0.0, 0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0, 1.0, 1.0],
                                    [2.0, 2.0, 2.0, 2.0, 2.0]]

partial
show : Property
show = fixedProperty $ do
  let x : Graph $ Tensor [] S32 = 1
  show x === "constant, shape=[], metadata={:0}"

  let x : Graph $ Tensor [] S32 = 1 + 2
  show x ===
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = tensor {dtype=F64} [1.3, 2.0, -0.4]
  show x === "constant, shape=[3], metadata={:0}"

partial
cast : Property
cast = property $ do
  shape <- forAll shapes

  lit <- forAll (literal shape nats)
  let x : Graph $ Tensor shape F64 = (do castDtype !(tensor {dtype=U32} lit))
  x ===# tensor (map (cast {to=Double}) lit)

  lit <- forAll (literal shape nats)
  let x : Graph $ Tensor shape F64 = (do castDtype !(tensor {dtype=U64} lit))
  x ===# tensor (map (cast {to=Double}) lit)

  lit <- forAll (literal shape int32s)
  let x : Graph $ Tensor shape F64 = (do castDtype !(tensor {dtype=S32} lit))
  x ===# tensor (map (cast {to=Double}) lit)

partial
identity : Property
identity = fixedProperty $ do
  identity ===# tensor {dtype=S32} []
  identity ===# tensor {dtype=S32} [[1]]
  identity ===# tensor {dtype=S32} [[1, 0], [0, 1]]
  identity ===# tensor {dtype=S32} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

  identity ===# tensor {dtype=F64} []
  identity ===# tensor {dtype=F64} [[1.0]]
  identity ===# tensor {dtype=F64} [[1.0, 0.0], [0.0, 1.0]]
  identity ===# tensor {dtype=F64} [
      [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    ]

namespace Vector
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = tensor {dtype=S32} [-2, 0, 1]
        r = tensor {dtype=S32} [3, 1, 2]
    l @@ r ===# -4

namespace Matrix
  export partial
  (@@) : Property
  (@@) = fixedProperty $ do
    let l = tensor {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = tensor {dtype=S32} [3, 3, -1]
    l @@ r ===# tensor [-7, 8]

    let l = tensor {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = tensor {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    l @@ r ===# tensor [[ -7,  -2], [  8, -11]]

partial
dotGeneral : Property
dotGeneral = fixedProperty $ do
  (do dotGeneral [] [] [] [] !2 !3) ===# tensor {dtype = S32} 6

  let l = fill {shape = [3, 4, 5, 6]} {dtype = S32} 1
      r = fill {shape = [3, 4, 6, 7]} 1
  -- contract on nothing
  (do dotGeneral [] [] [] [] !l !r) ===# fill 1
  (do dotGeneral [0] [0] [] [] !l !r) ===# fill 1
  (do dotGeneral [1] [1] [] [] !l !r) ===# fill 1
  (do dotGeneral [3] [2] [] [] !l !r) ===# fill 1
  (do dotGeneral [0, 1] [0, 1] [] [] !l !r) ===# fill 1
  (do dotGeneral [0, 3] [0, 2] [] [] !l !r) ===# fill 1
  (do dotGeneral [1, 3] [1, 2] [] [] !l !r) ===# fill 1
  (do dotGeneral [0, 1, 3] [0, 1, 2] [] [] !l !r) ===# fill 1

  -- contract on 4
  (do dotGeneral [0, 3] [0, 2] [1] [1] !l !r) ===# fill 4
  (do dotGeneral [0] [0] [1] [1] !l !r) ===# fill 4
  (do dotGeneral [] [] [1] [1] !l !r) ===# fill 4

  -- contract on 6
  (do dotGeneral [0, 1] [0, 1] [3] [2] !l !r) ===# fill 6
  (do dotGeneral [0] [0] [3] [2] !l !r) ===# fill 6
  (do dotGeneral [] [] [3] [2] !l !r) ===# fill 6

  -- contract on 3 and 6
  (do dotGeneral [1] [1] [0, 3] [0, 2] !l !r) ===# fill 18

  -- contract on 3, 4 and 6
  (do dotGeneral [] [] [0, 1, 3] [0, 1, 2] !l !r) ===# fill 72
  (do dotGeneral [] [] [3, 0, 1] [2, 0, 1] !l !r) ===# fill 72

  -- inputs generated with jax.random.uniform, expected generated with jax.lax.dot_general
  let l = tensor {dtype = F64} [[[0.64, 0.18, 0.02, 0.56],
                                 [0.55,  0.1, 0.34, 0.04],
                                 [0.09, 0.79, 0.35, 0.53]],
                                [[0.03, 0.42, 0.58, 0.91],
                                 [0.27, 0.15, 0.94, 0.52],
                                 [0.51, 0.91, 0.73, 0.96]]]
      r = tensor [[[0.53, 0.97],
                   [0.62, 0.87],
                   [0.63,  0.2],
                   [0.74, 0.85]],
                  [[0.22, 0.32],
                   [0.74, 0.79],
                   [0.37, 0.13],
                   [0.05, 0.61]]]
      expected = tensor [[[0.8778, 1.2574],
                          [0.5973, 0.7225],
                          [1.1502, 1.2951]],
                         [[0.5775, 0.9719],
                          [0.5442, 0.6443],
                          [1.1037, 1.5626]]]
  (do dotGeneral [0] [0] [2] [1] !l !r) ===# expected

partial
argmin : Property
argmin = property $ do
  d <- forAll dims
  xs <- forAll (literal [S d] doubles)
  let xs = tensor xs
  (do slice [at !(argmin !xs)] !xs) ===# (do reduce [0] @{Min} !xs)

partial
argmax : Property
argmax = property $ do
  d <- forAll dims
  xs <- forAll (literal [S d] doubles)
  let xs = tensor xs
  (do slice [at !(argmax !xs)] !xs) ===# (do reduce [0] @{Max} !xs)

partial
select : Property
select = fixedProperty $ do
  let onTrue = tensor {dtype=S32} 1
      onFalse = tensor 0
  (do select !(tensor True) !onTrue !onFalse) ===# onTrue
  (do select !(tensor False) !onTrue !onFalse) ===# onFalse

  let pred = tensor [[False, True, True], [True, False, False]]
      onTrue = tensor {dtype=S32} [[0, 1, 2], [3, 4, 5]]
      onFalse = tensor [[6, 7, 8], [9, 10, 11]]
      expected = tensor [[6, 1, 2], [3, 10, 11]]
  (do select !pred !onTrue !onFalse) ===# expected

partial
erf : Property
erf = fixedProperty $ do
  let x = tensor [-1.5, -0.5, 0.5, 1.5]
      expected = tensor [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  (do erf !x) ===# expected

partial
cholesky : Property
cholesky = fixedProperty $ do
  let x = tensor [[1.0, 0.0], [2.0, 0.0]]
      expected = tensor [[nan, 0], [nan, nan]]
  (do cholesky !x) ===# expected

  -- example generated with tensorflow
  let x = tensor [
              [ 2.236123  ,  0.70387983,  2.8447943 ],
              [ 0.7059226 ,  2.661426  , -0.8714733 ],
              [ 1.3730898 ,  1.4064665 ,  2.7474475 ]
            ]
      expected = tensor [
              [1.4953672 , 0.0       , 0.0       ],
              [0.47207308, 1.5615932 , 0.0       ],
              [0.9182292 , 0.6230785 , 1.2312902 ]
            ]
  (do cholesky !x) ===# expected

partial
triangularSolveResultAndInverse : Property
triangularSolveResultAndInverse = fixedProperty $ do
  let a = tensor [
              [0.8578532 , 0.0       , 0.0       ],
              [0.2481904 , 0.9885198 , 0.0       ],
              [0.59390426, 0.14998078, 0.19468737]
            ]
      b = tensor [
              [0.45312142, 0.37276268],
              [0.9210588 , 0.00647926],
              [0.7890165 , 0.77121615]
            ]
      actual = a |\ b
      expected = tensor [
                    [ 0.52820396,  0.43452972],
                    [ 0.79913783, -0.10254406],
                    [ 1.8257918 ,  2.7147462 ]
                  ]
  actual ===# expected
  a @@ actual ===# b

  let actual = a.T \| b
      expected = tensor [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  actual ===# expected
  a.T @@ actual ===# b

partial
triangularSolveIgnoresOppositeElems : Property
triangularSolveIgnoresOppositeElems = fixedProperty $ do
  let a = tensor [[1.0, 2.0], [3.0, 4.0]]
      aLower = tensor [[1.0, 0.0], [3.0, 4.0]]
      b = tensor [5.0, 6.0]
  a |\ b ===# aLower |\ b

  let aUpper = tensor [[1.0, 2.0], [0.0, 4.0]]
  a \| b ===# aUpper \| b

partial
trace : Property
trace = fixedProperty $ do
  let x = tensor {dtype=S32} [[-1, 5], [1, 4]]
  (do trace !x) ===# 3

export partial
group : Group
group = MkGroup "Tensor" $ [
      ("eval . tensor", tensorThenEval)
    , ("eval multiple tensors", evalTuple)
    , ("eval multiple tensors for non-trivial graph", evalTupleNonTrivial)
    , ("can read/write finite numeric bounds to/from XLA", canConvertAtXlaNumericBounds)
    , ("bounded non-finite", boundedNonFinite)
    , ("iota", iota)
    , ("iota examples", iotaExamples)
    , ("show", show)
    , ("cast", cast)
    , ("identity", identity)
    , ("Vector.(@@)", Vector.(@@))
    , ("Matrix.(@@)", Matrix.(@@))
    , ("dotGeneral", dotGeneral)
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
