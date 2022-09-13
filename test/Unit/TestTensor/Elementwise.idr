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
module Unit.TestTensor.Elementwise

import System

import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

namespace S32
  export partial
  testElementwiseUnary :
    (Int32 -> Int32) ->
    (forall shape . Tensor shape S32 -> Ref $ Tensor shape S32) ->
    Property
  testElementwiseUnary fInt fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape int32s)
    let x' = tensor x
    [| fInt x |] === unsafeEval (do fTensor !x')

namespace F64
  export partial
  testElementwiseUnary :
    (Double -> Double) ->
    (forall shape . Tensor shape F64 -> Ref $ Tensor shape F64) ->
    Property
  testElementwiseUnary fDouble fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape doubles)
    let x' = tensor x
    [| fDouble x |] ==~ unsafeEval (do fTensor !x')

namespace PRED
  export partial
  testElementwiseUnary :
    (Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Ref $ Tensor shape PRED) ->
    Property
  testElementwiseUnary fBool fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape bool)
    let x' = tensor x
    [| fBool x |] === unsafeEval (do fTensor !x')

namespace S32
  export partial
  testElementwiseBinary :
    (Int32 -> Int32 -> Int32) ->
    (forall shape . Ref (Tensor shape S32) -> Ref (Tensor shape S32) -> Ref (Tensor shape S32)) ->
    Property
  testElementwiseBinary fInt fTensor = property $ do
    shape <- forAll shapes
    let int32s = literal shape int32s
    [x, y] <- forAll (np [int32s, int32s])
    let x' = tensor {dtype=S32} x
        y' = tensor {dtype=S32} y
    [| fInt x y |] === unsafeEval (fTensor x' y')

namespace U64
  export partial
  testElementwiseBinarySuccRight :
    (Nat -> Nat -> Nat) ->
    (forall shape . Ref (Tensor shape U64) -> Ref (Tensor shape U64) -> Ref (Tensor shape U64)) ->
    Property
  testElementwiseBinarySuccRight fInt fTensor = property $ do
    shape <- forAll shapes
    let nats = literal shape nats
    [x, y] <- forAll (np [nats, map (map S) nats])
    let x' = tensor {dtype=U64} x
        y' = tensor {dtype=U64} y
    [| fInt x y |] === unsafeEval (fTensor x' y')

namespace F64
  export partial
  testElementwiseBinary :
    (Double -> Double -> Double) ->
    (forall shape . Ref (Tensor shape F64) -> Ref (Tensor shape F64) -> Ref (Tensor shape F64)) ->
    Property
  testElementwiseBinary fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = tensor {dtype=F64} x
        y' = tensor {dtype=F64} y
    [| fDouble x y |] ==~ unsafeEval (fTensor x' y')

namespace PRED
  export partial
  testElementwiseBinary :
    (Bool -> Bool -> Bool) ->
    (forall shape . Ref (Tensor shape PRED) -> Ref (Tensor shape PRED) -> Ref (Tensor shape PRED)) ->
    Property
  testElementwiseBinary fBool fTensor = property $ do
    shape <- forAll shapes
    let bools = literal shape bool
    [x, y] <- forAll (np [bools, bools])
    let x' = tensor {dtype=PRED} x
        y' = tensor {dtype=PRED} y
    [| fBool x y |] === unsafeEval (fTensor x' y')

partial
scalarMultiplication : Property
scalarMultiplication = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = tensor {dtype=F64} lit
          scalar' = tensor {dtype=F64} (Scalar scalar)
      map (scalar *) lit ==~ unsafeEval (scalar' * lit')

partial
scalarDivision : Property
scalarDivision = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = tensor {dtype=F64} lit
          scalar' = tensor {dtype=F64} (Scalar scalar)
      map (/ scalar) lit ==~ unsafeEval (lit' / scalar')

namespace S32
  export partial
  testElementwiseComparator :
    (Int32 -> Int32 -> Bool) ->
    (forall shape . Ref (Tensor shape S32) -> Ref (Tensor shape S32) -> Ref (Tensor shape PRED)) ->
    Property
  testElementwiseComparator fInt fTensor = property $ do
    shape <- forAll shapes
    let int32s = literal shape int32s
    [x, y] <- forAll (np [int32s, int32s])
    let x' = tensor {dtype=S32} x
        y' = tensor {dtype=S32} y
    [| fInt x y |] === unsafeEval (fTensor x' y')

namespace F64
  export partial
  testElementwiseComparator :
    (Double -> Double -> Bool) ->
    (forall shape . Ref (Tensor shape F64) -> Ref (Tensor shape F64) -> Ref (Tensor shape PRED)) ->
    Property
  testElementwiseComparator fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = tensor {dtype=F64} x
        y' = tensor {dtype=F64} y
    [| fDouble x y |] === unsafeEval (fTensor x' y')

namespace PRED
  export partial
  testElementwiseComparator :
    (Bool -> Bool -> Bool) ->
    (forall shape . Ref (Tensor shape PRED) -> Ref (Tensor shape PRED) -> Ref (Tensor shape PRED)) ->
    Property
  testElementwiseComparator = testElementwiseBinary

partial
neutralIsNeutralForSum : Property
neutralIsNeutralForSum = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = tensor {dtype=F64} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  unsafeEval right ==~ x
  unsafeEval left ==~ x

  x <- forAll (literal shape int32s)
  let x' = tensor {dtype=S32} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  unsafeEval right === x
  unsafeEval left === x

partial
neutralIsNeutralForProd : Property
neutralIsNeutralForProd = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = tensor {dtype=F64} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  unsafeEval right ==~ x
  unsafeEval left ==~ x

  x <- forAll (literal shape int32s)
  let x' = tensor {dtype=S32} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  unsafeEval right === x
  unsafeEval left === x

partial
neutralIsNeutralForAny : Property
neutralIsNeutralForAny = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = tensor {dtype=PRED} x
      right = (<+>) @{Any} x' (neutral @{Monoid.Any})
      left = (<+>) @{Any} (neutral @{Monoid.Any}) x'
  unsafeEval right === x
  unsafeEval left === x

partial
neutralIsNeutralForAll : Property
neutralIsNeutralForAll = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = tensor {dtype=PRED} x
      right = (<+>) @{All} x' (neutral @{Monoid.All})
      left = (<+>) @{All} (neutral @{Monoid.All}) x'
  unsafeEval right === x
  unsafeEval left === x

partial
neutralIsNeutralForMin : Property
neutralIsNeutralForMin = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = tensor {dtype=F64} x
      right = (<+>) @{Min} x' (neutral @{Min})
      left = (<+>) @{Min} (neutral @{Min}) x'
  unsafeEval right ==~ x
  unsafeEval left ==~ x

partial
neutralIsNeutralForMax : Property
neutralIsNeutralForMax = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = tensor {dtype=F64} x
      right = (<+>) @{Max} x' (neutral @{Max})
      left = (<+>) @{Max} (neutral @{Max}) x'
  unsafeEval right ==~ x
  unsafeEval left ==~ x

min' : Double -> Double -> Double
min' x y = if x == x && y == y then min x y else nan

max' : Double -> Double -> Double
max' x y = if x == x && y == y then max x y else nan

and : Bool -> Bool -> Bool
and x y = x && y

or : Bool -> Bool -> Bool
or x y = x || y

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

export partial
all : List (PropertyName, Property)
all = [
      ("negate S32", S32.testElementwiseUnary negate (Tensor.negate . pure))
    , ("negate F64", F64.testElementwiseUnary negate (Tensor.negate . pure))
    , ("recip", F64.testElementwiseUnary recip recip)
    , ("abs S32", S32.testElementwiseUnary abs abs)
    , ("abs F64", F64.testElementwiseUnary abs abs)
    , ("exp", F64.testElementwiseUnary exp exp)
    , ("ceil", F64.testElementwiseUnary ceiling ceil)
    , ("floor", F64.testElementwiseUnary floor floor)
    , ("log", F64.testElementwiseUnary log log)
    , ("logistic", F64.testElementwiseUnary (\x => 1 / (1 + exp (-x))) logistic)
    , ("sin", F64.testElementwiseUnary sin sin)
    , ("cos", F64.testElementwiseUnary cos cos)
    , ("tan", F64.testElementwiseUnary tan tan)
    , ("asin", F64.testElementwiseUnary asin asin)
    , ("acos", F64.testElementwiseUnary acos acos)
    , ("atan", F64.testElementwiseUnary atan atan)
    , ("sinh", F64.testElementwiseUnary sinh sinh)
    , ("cosh", F64.testElementwiseUnary cosh cosh)
    , ("tanh", F64.testElementwiseUnary tanh' tanh)
    , ("asinh", F64.testElementwiseUnary asinh Tensor.asinh)
    , ("acosh", F64.testElementwiseUnary acosh Tensor.acosh)
    , ("atanh", F64.testElementwiseUnary atanh Tensor.atanh)
    , ("sqrt", F64.testElementwiseUnary sqrt sqrt)
    , ("square", F64.testElementwiseUnary (\x => x * x) square)
    , ("not", PRED.testElementwiseUnary not not)

    , ("(+) F64", F64.testElementwiseBinary (+) (+))
    , ("(+) S32", S32.testElementwiseBinary (+) (+))
    , ("(-) F64", F64.testElementwiseBinary (-) (-))
    , ("(-) S32", S32.testElementwiseBinary (-) (-))
    , ("(*) F64", F64.testElementwiseBinary (*) (*))
    , ("(*) S32", S32.testElementwiseBinary (*) (*))
    , ("(/)", F64.testElementwiseBinary (/) (/))
    -- , ("pow", F64.testElementwiseBinary pow (^)),  bug in idris 0.5.1 for pow
    , ("min S32", S32.testElementwiseBinary min (\x, y => min !x !y))
    , ("max S32", S32.testElementwiseBinary max (\x, y => max !x !y))
    , ("min F64", F64.testElementwiseBinary min' (\x, y => min !x !y))
    , ("max F64", F64.testElementwiseBinary max' (\x, y => max !x !y))
    , ("(&&)", PRED.testElementwiseBinary and (&&))
    , ("(||)", PRED.testElementwiseBinary or (||))

    , ("Scalarwise.(*)", scalarMultiplication)
    , ("Scalarwise.(/)", scalarDivision)

    , ("div", U64.testElementwiseBinarySuccRight div div)
    , ("rem", U64.testElementwiseBinarySuccRight mod Tensor.rem)

    , ("(==) F64", F64.testElementwiseComparator (==) (==))
    , ("(==) S32", S32.testElementwiseComparator (==) (==))
    , ("(==) PRED", PRED.testElementwiseComparator (==) (==))
    , ("(/=) F64", F64.testElementwiseComparator (/=) (/=))
    , ("(/=) S32", S32.testElementwiseComparator (/=) (/=))
    , ("(/=) PRED", PRED.testElementwiseComparator (/=) (/=))
    , ("(<) F64", F64.testElementwiseComparator (<) (<))
    , ("(<) S32", S32.testElementwiseComparator (<) (<))
    , ("(>) F64", F64.testElementwiseComparator (>) (>))
    , ("(>) S32", S32.testElementwiseComparator (>) (>))
    , ("(<=) F64", F64.testElementwiseComparator (<=) (<=))
    , ("(<=) S32", S32.testElementwiseComparator (<=) (<=))
    , ("(>=) F64", F64.testElementwiseComparator (>=) (>=))
    , ("(>=) S32", S32.testElementwiseComparator (>=) (>=))

    , ("Sum", neutralIsNeutralForSum)
    , ("Prod", neutralIsNeutralForProd)
    , ("Min", neutralIsNeutralForMin)
    , ("Max", neutralIsNeutralForMax)
    , ("Any", neutralIsNeutralForAny)
    , ("All", neutralIsNeutralForAll)
  ]
