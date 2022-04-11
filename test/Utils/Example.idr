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
module Utils.Example

import System

import Data.Hashable

import Literal
import Tensor

export
isNan : Double -> Bool
isNan x = x /= x

export
assert : String -> Bool -> IO ()
assert name x = unless x $ do
  putStrLn ("Test failed: " ++ name)
  exitFailure

export
bools : List (Literal [] Bool)
bools = [True, False]

namespace Literal
  export
  ints : List (Literal [] Int)
  ints = [-3, -1, 0, 1, 3]

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

namespace Double
  export
  doubles : List Double
  doubles = [-inf, -3.4, -1.1, -0.1, 0.0, 0.1, 1.1, 3.4, inf, nan]

namespace Literal
  export
  doubles : List (Literal [] Double)
  doubles = map Scalar doubles

export
floatingPointTolerance : Double
floatingPointTolerance = 0.00000001

namespace Double
  export
  sufficientlyEq : {default floatingPointTolerance tol : Double} -> Double -> Double -> Bool
  sufficientlyEq x y =  -- moved
    x /= x && y /= y  -- nan
    || x == y  -- inf
    -- `let` avoids compiler bug
    || (let diff = abs (x - y) in diff < tol)  -- real

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
  test_sufficientlyEq : IO ()
  test_sufficientlyEq = do
    sequence_ [assert "sufficientlyEq for suff. equal" $ sufficientlyEq x y
                | (x, y) <- sufficientlyEqCases]
    sequence_ [assert "sufficientlyEq for insuff. equal" $ not (sufficientlyEq x y)
                | (x, y) <- insufficientlyEqCases]

export
test : IO ()
test = do
  Double.test_sufficientlyEq
