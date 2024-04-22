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

import Control.Monad.Either
import System

import Data.SOP
import Hedgehog

import Device
import Types

import TestUtils
import Utils.TestComparison
import Unit.Model.TestKernel
import Unit.TestDistribution
import Unit.TestTensor
import Unit.TestLiteral
import Unit.TestUtil

import PjrtCpuPlugin
import PjrtGpuPlugin
-- bad import
import Compiler.Xla.PJRT.C.PJRT_C_API

partial
main : IO ()
main = do
  Right device <- runEitherT $ do device !PjrtCpuPlugin.getPjrtApi
    | Left err => die $ show err

  test [
      Utils.TestComparison.group
    , TestUtils.group
    , Unit.TestUtil.group
    , Unit.TestLiteral.group
    , Unit.TestTensor.group
    , Unit.TestDistribution.group
    , Unit.Model.TestKernel.group
  ]
