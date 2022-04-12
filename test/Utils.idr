{--
Copyright 2022 Joel Berkeley

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
module Utils

import public Data.SOP
import Data.Bounded
import public Hedgehog

import Literal
import Tensor
import Types

import Utils.Example

export
isNan : Double -> Bool
isNan x = x /= x

namespace Double
  export
  nan, inf : Double
  nan = 0.0 / 0.0
  inf = 1.0 / 0.0

namespace Literal
  export
  nan, inf : Literal [] Double
  nan = Scalar (0.0 / 0.0)
  inf = Scalar (1.0 / 0.0)

export
floatingPointTolerance : Double
floatingPointTolerance = 0.00000001

namespace Double
  export
  sufficientlyEq : {default floatingPointTolerance tol : Double} -> Double -> Double -> Bool
  sufficientlyEq x y =  -- moved
    x /= x && y /= y  -- nan
    || x == y  -- inf
    || abs (x - y) < tol  -- real

namespace Literal
  export
  sufficientlyEq : {default floatingPointTolerance tol : Double} -> {shape : _}
                   -> Tensor shape F64 -> Tensor shape F64 -> Bool
  sufficientlyEq x y = all [| sufficientlyEq {tol} (toLiteral x) (toLiteral y) |]

sufficientlyEqCases : List (Double, Double)
sufficientlyEqCases = [
  (0.0, 0.0),
  (0.0, floatingPointTolerance / 2),
  (floatingPointTolerance / 2, 0.0),
  (0.0, - floatingPointTolerance / 2),
  (- floatingPointTolerance / 2, 0.0),
  (1.1, 1.1),
  (1.1, 1.1 + floatingPointTolerance / 2),
  (1.1, 1.1 - floatingPointTolerance / 2),
  (-1.1, -1.1),
  (-1.1, -1.1 + floatingPointTolerance / 2),
  (-1.1, -1.1 - floatingPointTolerance / 2),
  (inf, inf),
  (-inf, -inf),
  (nan, nan)
]

insufficientlyEqCases : List (Double, Double)
insufficientlyEqCases =
  let cases = [
      (0.0, floatingPointTolerance * 2),
      (0.0, - floatingPointTolerance * 2),
      (1.1, 1.1 + floatingPointTolerance * 2),
      (1.1, 1.1 - floatingPointTolerance * 2),
      (-1.1, -1.1 + floatingPointTolerance * 2),
      (-1.1, -1.1 - floatingPointTolerance * 2),
      (0.0, inf),
      (1.1, inf),
      (-1.1, inf),
      (0.0, -inf),
      (1.1, -inf),
      (-1.1, -inf),
      (0.0, nan),
      (1.1, nan),
      (-1.1, nan),
      (inf, -inf),
      (inf, nan),
      (-inf, nan)
  ] in cases ++ map (\(x, y) => (y, x)) cases

namespace Double
  export
  test_sufficientlyEq : Property
  test_sufficientlyEq = withTests 1 $ property $ do
    traverse_ (\(x, y) => sufficientlyEq x y === True) sufficientlyEqCases
    traverse_ (\(x, y) => (sufficientlyEq x y) === False) insufficientlyEqCases

export
group : Group
group = MkGroup "Test utilities" [
    ("sufficientlyEq", Double.test_sufficientlyEq)
  ]

maxRank : Nat
maxRank = 5

maxDim : Nat
maxDim = 10

export
shapes : Gen Shape
shapes = list (linear 0 maxRank) (nat $ linear 0 maxDim)

export covering
literal : (shape : Shape) -> Gen a -> Gen (Literal shape a)
literal [] gen = map Scalar gen
literal (0 :: _) gen = pure []
literal (S d :: ds) gen = [| literal ds gen :: literal (d :: ds) gen |]

pow : Prelude.Num ty => ty -> Nat -> ty
pow x Z = x
pow x (S k) = x * pow x k

intBound : Int
intBound = pow 2 10

export
ints : Gen Int
ints = int $ linear (-intBound) intBound

doubleBound : Double
doubleBound = 9999

numericDoubles : Gen Double
numericDoubles = double $ exponentialDoubleFrom (-doubleBound) 0 doubleBound

export
doubles : Gen Double
doubles = frequency [(1, numericDoubles), (3, element [-inf, inf, nan])]

export
doublesWithoutNan : Gen Double
doublesWithoutNan = frequency [(1, numericDoubles), (3, element [-inf, inf])]

infix 1 ==~

export covering
(==~) : Monad m => {default floatingPointTolerance tol : Double} -> {shape : _} ->
        Literal shape Double -> Literal shape Double -> TestT m ()
(==~) x y = diff x sufficientlyEq' y
  where
  sufficientlyEq' : {shape : _} -> Literal shape Double -> Literal shape Double -> Bool
  sufficientlyEq' x y = all [| sufficientlyEq {tol} x y |]

infix 1 ===?

namespace S32
  export
  (===?) : Monad m => {shape : _} -> Tensor shape S32 -> Tensor shape S32 -> TestT m ()
  x ===? y = (toLiteral x) === (toLiteral y)

namespace PRED
  export
  (===?) : Monad m => {shape : _} -> Tensor shape PRED -> Tensor shape PRED -> TestT m ()
  x ===? y = (toLiteral x) === (toLiteral y)

export
fpTensorEq : Monad m => {default floatingPointTolerance tol : Double} -> {shape : _} ->
             Tensor shape F64 -> Tensor shape F64 -> TestT m ()
fpTensorEq x y = (toLiteral x) ==~ (toLiteral y)

namespace F64
  export
  (===?) : Monad m => {shape : _} -> Tensor shape F64 -> Tensor shape F64 -> TestT m ()
  (===?) = fpTensorEq
