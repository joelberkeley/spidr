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
module Unit.TestTensor.AD

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

secondDerivative : Device => Property
secondDerivative @{device} = fixedProperty $ do
  let cube : Tensor [] F64 -> Tensor [] F64
      cube x = x * x * x

  (let x = grad (grad (pure . cube)) (tensor 3.0) in unsafePerformIO $ do putStrLn (show x); eval device x) === 18.0

--  let f : Tensor [] F64 -> Tag $ Tensor [] F64
--      f x = reduce @{Sum} [0] $ broadcast {to = [3]} x
--
--  grad f (tensor 7.0) ===# pure (tensor 3.0)

export
all : Device => List (PropertyName, Property)
all = [
    --  ("first derivatives", firstDerivative)
     ("second derivatives", secondDerivative)
  ]
