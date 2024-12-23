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

square : Device => Property
square = fixedProperty $ do
  sqrt (square $ tensor 3.0) ===# tensor 3.0
--  grad (pure . square) (tensor 1.0) ===# pure (tensor 2.0)
{-
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = tensor {dtype = F64} x
  map id x ==~ unsafeEval (map pure x')
  map (1.0 /) x ==~ Tag.unsafeEval (map (pure . (1.0 /)) x')
-}

export
all : Device => List (PropertyName, Property)
all = [
      ("grad square", square)
  ]
