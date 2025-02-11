{--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
