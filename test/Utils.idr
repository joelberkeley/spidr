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

import System

import Tensor

export
assert : String -> Bool -> IO ()
assert name x = unless x $ do
    putStrLn ("Test failed: " ++ name)
    exitFailure

export
assertAll : String -> {shape : _} -> Tensor shape PRED -> IO ()
assertAll name xs = assert name (arrayAll !(eval xs)) where
    arrayAll : {shape : _} -> Array shape Bool -> Bool
    arrayAll {shape = []} x = x
    arrayAll {shape = (0 :: _)} [] = True
    arrayAll {shape = ((S d) :: ds)} (x :: xs) = arrayAll x && arrayAll {shape=(d :: ds)} xs

export
bools : List Bool
bools = [True, False]

export
ints : List Int
ints = [-3, -1, 0, 1, 3]

export
nan, inf : Double
nan = 0.0 / 0.0
inf = 1.0 / 0.0

export
doubles : List Double
doubles = [-inf, -3.4, -1.1, -0.1, 0.0, 0.1, 1.1, 3.4, inf, nan]

export
floatingPointTolerance : Double
floatingPointTolerance = 0.00000001

export
sufficientlyEq : Double -> Double -> Bool
sufficientlyEq x y =
    x /= x && y /= y  -- nan
    || x == y  -- inf
    || abs (x - y) < floatingPointTolerance  -- real

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

export
test_sufficientlyEq : IO ()
test_sufficientlyEq = do
    sequence_ [assert "sufficientlyEq for suff. equal" $ sufficientlyEq x y
               | (x, y) <- sufficientlyEqCases]
    sequence_ [assert "sufficientlyEq for insuff. equal" $ not (sufficientlyEq x y)
               | (x, y) <- insufficientlyEqCases]

-- WARNING: This uses a number of functions, and thus assumes they work, so
-- we shouldn't use it to test them.
export
sufficientlyEqEach : {shape : _} -> Tensor shape F64 -> Tensor shape F64 -> Tensor shape PRED
sufficientlyEqEach x y =
    x /=# x &&# y /=# y  -- nan
    ||# x ==# y  -- inf
    ||# absEach (x - y) <# fill floatingPointTolerance  -- real

export
test_sufficientlyEqEach : IO ()
test_sufficientlyEqEach = do
    let x = const [[0.0, 1.1, inf], [-inf, nan, -1.1]]
        y = const [[0.1, 1.1, inf], [inf, nan, 1.1]]
    eq <- eval {shape=[_, _]} (sufficientlyEqEach x y)
    assert "sufficientlyEqEach for array" (eq == [[False, True, True], [False, True, False]])

    sequence_ [assertAll "sufficientlyEq for suff. equal scalars" $
               sufficientlyEqEach {shape=[]} (const x) (const y)
               | (x, y) <- sufficientlyEqCases]
    sequence_ [assertAll "sufficientlyEq for suff. equal scalars" $
               notEach (sufficientlyEqEach {shape=[]} (const x) (const y))
               | (x, y) <- insufficientlyEqCases]
