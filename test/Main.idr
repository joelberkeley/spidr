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

import Control.Monad.Maybe
import Data.SOP
import Hedgehog

import Device

import TestUtils
import Utils.TestComparison
import Unit.Model.TestKernel
import Unit.TestDistribution
import Unit.TestTensor
import Unit.TestLiteral
import Unit.TestUtil

import System

partial
main : IO ()
main = do
  -- we need to check this works for
  -- cpu binary with cpu and gpu handles
  -- gpu binary with cpu and gpu handles, and with and without access to gpu (use docker without --gpus all for that)
  Just gpu <- runMaybeT cuda | Nothing => die "no gpu"
  test [
      Utils.TestComparison.group
    , TestUtils.group
    , Unit.TestUtil.group
    , Unit.TestLiteral.group
    , Unit.TestTensor.group
    , Unit.TestDistribution.group
    , Unit.Model.TestKernel.group
  ]
