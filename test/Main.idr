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
module Main

import Data.SOP
import Hedgehog

import Unit.Model.TestKernel
import Unit.Util.TestHashable
import Unit.TestDistribution
import Unit.TestTensor
import Unit.TestLiteral
import Unit.TestUtil

import Utils

main : IO ()
main = do
  Utils.test

  Unit.TestUtil.test
  Unit.Util.TestHashable.test
  Unit.TestLiteral.test
  Unit.TestTensor.test
  Unit.Model.TestKernel.test
  Unit.TestDistribution.test

  putStrLn "Old style tests passed\n\n"

  _ <- checkGroup $ MkGroup "All" [
    ("Scalar addition", scalarAddition),
    ("Vector addition", vectorAddition)
    -- ("arrayAddition", arrayAddition),
    -- ("test_addition", test_addition)
  ]
