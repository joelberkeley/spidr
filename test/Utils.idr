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
floatingPointTolerance : Double
floatingPointTolerance = 0.00000001

isNan : Double -> Bool
isNan x = x /= x

nan, inf : Double
nan = 0.0 / 0.0
inf = 1.0 / 0.0

export
doubleSufficientlyEq : Double -> Double -> Bool
doubleSufficientlyEq x y =
    if isNan x || isNan y then isNan x && isNan y else
    if x == inf || x == -inf || y == inf || y == -inf then x == y else
    abs (x - y) < floatingPointTolerance

export
assert : Bool -> IO ()
assert x = unless x $ do
    putStrLn "Test failed"
    exitFailure

export
assertAll : {shape : _} -> Tensor shape Bool -> IO ()
assertAll xs = assert (arrayAll !(eval xs)) where
    arrayAll : {shape : _} -> Array shape {dtype=Bool} -> Bool
    arrayAll {shape = []} x = x
    arrayAll {shape = (0 :: _)} [] = True
    arrayAll {shape = ((S d) :: ds)} (x :: xs) = arrayAll x && arrayAll {shape=(d :: ds)} xs

isNanEach : Tensor shape Double -> Tensor shape Bool
isNanEach x = x /=# x

-- WARNING: This uses a number of functions, and thus assumes they work, so
-- we shouldn't use it to test them.
export
fpEq : {shape : _} -> Tensor shape Double -> Tensor shape Double -> Tensor shape Bool
fpEq x y =
    let either_nan = isNanEach x ||# isNanEach y
        both_nan = isNanEach x &&# isNanEach y
        either_inf =
            x ==# fill inf
            ||# x ==# fill (-inf)
            ||# y ==# fill inf
            ||# y ==# fill (-inf) in
    (either_nan &&# both_nan)
    ||# (either_inf &&# x ==# y)
    ||# (absE (x - y) <# fill floatingPointTolerance) where

export
bools : List Bool
bools = [True, False]

export
ints : List Int
ints = [-3, -1, 0, 1, 3]

export
doubles : List Double
doubles = [-inf, -3.4, -1.1, -0.1, 0.0, 0.1, 1.1, 3.4, inf, nan]
