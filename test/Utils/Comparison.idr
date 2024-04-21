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
module Utils.Comparison

import public Data.SOP
import Data.Bounded
import public Hedgehog

import Device
import Literal
import Tensor

import Utils

export
floatingPointTolerance : Double
floatingPointTolerance = 0.000001

export
sufficientlyEq : {default floatingPointTolerance tol : Double} -> Double -> Double -> Bool
sufficientlyEq x y =
  x /= x && y /= y  -- nan
  || x == y  -- inf
  || let diff = x - y
         avg = (x + y) / 2
      in abs (diff / avg) < tol  -- real

infix 1 ==~

export covering
(==~) :
  Monad m =>
  {default floatingPointTolerance tol : Double} ->
  {shape : _} ->
  Literal shape Double ->
  Literal shape Double ->
  TestT m ()
(==~) x y = diff x sufficientlyEq' y
  where
  sufficientlyEq' : {shape : _} -> Literal shape Double -> Literal shape Double -> Bool
  sufficientlyEq' x y = all [| sufficientlyEq {tol} x y |]

infix 1 ===#

namespace PRED
  export partial
  (===#) : Device => Monad m => {shape : _} -> Graph (Tensor shape PRED) -> Graph (Tensor shape PRED) -> TestT m ()
  x ===# y = unsafeEval x === unsafeEval y

namespace S32
  export partial
  (===#) : Device => Monad m => {shape : _} -> Graph (Tensor shape S32) -> Graph (Tensor shape S32) -> TestT m ()
  x ===# y = unsafeEval x === unsafeEval y

namespace U32
  export partial
  (===#) : Device => Monad m => {shape : _} -> Graph (Tensor shape U32) -> Graph (Tensor shape U32) -> TestT m ()
  x ===# y = unsafeEval x === unsafeEval y

namespace U64
  export partial
  (===#) : Device => Monad m => {shape : _} -> Graph (Tensor shape U64) -> Graph (Tensor shape U64) -> TestT m ()
  x ===# y = unsafeEval x === unsafeEval y

namespace F64
  export partial
  (===#) : Device => Monad m => {shape : _} -> {default floatingPointTolerance tol : Double} ->
           Graph (Tensor shape F64) -> Graph (Tensor shape F64) -> TestT m ()
  x ===# y = (==~) {tol} (unsafeEval x) (unsafeEval y)
