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

export infix 1 ==~

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

export infix 6 ===#

namespace Tag
  namespace PRED
    export
    (===#) : Device =>
             Monad m =>
             {shape : _} ->
             Tag (Tensor shape PRED) ->
             Tag (Tensor shape PRED) ->
             TestT m ()
    x ===# y = unsafeEval x === unsafeEval y

  namespace S32
    export
    (===#) : Device =>
             Monad m =>
             {shape : _} ->
             Tag (Tensor shape S32) ->
             Tag (Tensor shape S32) ->
             TestT m ()
    x ===# y = unsafeEval x === unsafeEval y

  namespace U32
    export
    (===#) : Device =>
             Monad m =>
             {shape : _} ->
             Tag (Tensor shape U32) ->
             Tag (Tensor shape U32) ->
             TestT m ()
    x ===# y = unsafeEval x === unsafeEval y

  namespace U64
    export
    (===#) : Device =>
             Monad m =>
             {shape : _} ->
             Tag (Tensor shape U64) ->
             Tag (Tensor shape U64) ->
             TestT m ()
    x ===# y = unsafeEval x === unsafeEval y

  namespace F64
    export
    (===#) : Device =>
             Monad m =>
             {shape : _} ->
             {default floatingPointTolerance tol : Double} ->
             Tag (Tensor shape F64) ->
             Tag (Tensor shape F64) ->
             TestT m ()
    x ===# y = (==~) {tol} (unsafeEval x) (unsafeEval y)

namespace PRED
  export
  (===#) : Device => Monad m => {shape : _} -> Tensor shape PRED -> Tensor shape PRED -> TestT m ()
  x ===# y = unsafeEval (pure x) === unsafeEval (pure y)

namespace S32
  export
  (===#) : Device => Monad m => {shape : _} -> Tensor shape S32 -> Tensor shape S32 -> TestT m ()
  x ===# y = unsafeEval (pure x) === unsafeEval (pure y)

namespace U32
  export
  (===#) : Device => Monad m => {shape : _} -> Tensor shape U32 -> Tensor shape U32 -> TestT m ()
  x ===# y = unsafeEval (pure x) === unsafeEval (pure y)

namespace U64
  export
  (===#) : Device => Monad m => {shape : _} -> Tensor shape U64 -> Tensor shape U64 -> TestT m ()
  x ===# y = unsafeEval (pure x) === unsafeEval (pure y)

namespace F64
  export
  (===#) : Device => Monad m => {shape : _} -> {default floatingPointTolerance tol : Double} ->
           Tensor shape F64 -> Tensor shape F64 -> TestT m ()
  x ===# y = (==~) {tol} (unsafeEval $ pure x) (unsafeEval $ pure y)
