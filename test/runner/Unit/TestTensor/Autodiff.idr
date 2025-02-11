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
module Unit.TestTensor.Autodiff

import System

import Device
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

firstDerivative : Device => Property
firstDerivative = fixedProperty $ do
  let f : Tensor [] F64 -> Tag $ Tensor [] F64
      f x = do
        x <- tag $ x + x
        pure (x * x)

  grad f (tensor 3.0) ===# pure (tensor 24.0)

  let f : Tensor [2] F64 -> Tag $ Tensor [] F64
      f x = pure $ slice [at 0] $ square x

  grad f (tensor [3.0, 5.0]) ===# pure (tensor [6.0, 0.0])

  let f : Tensor [2] F64 -> Tag $ Tensor [] F64
      f x = reduce @{Sum} [0] (square x)

  grad f (tensor [3.14, 2.72]) ===# pure (tensor [6.28, 5.44])

higherDerivatives : Device => Property
higherDerivatives @{device} = fixedProperty $ do
  let quartic : Tensor [] F64 -> Tensor [] F64
      quartic x = x * x * x * x

  grad (grad (pure . quartic)) (tensor 3.0) ===# pure (tensor 108.0)
  grad (grad (grad (pure . quartic))) (tensor 0.1) ===# pure (tensor 2.4)

export
all : Device => List (PropertyName, Property)
all = [
      ("first derivatives", firstDerivative)
    , ("higher derivatives", higherDerivatives)
  ]
