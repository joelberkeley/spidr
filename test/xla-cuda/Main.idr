{--
Copyright 2024 Joel Berkeley

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

import System

import TestRunner
import PjrtPluginXlaCuda

import Control.Linear.LIO
import Control.Monad.State
import Data.Linear.Notation

import Compiler.Xla.XlaData
import Compiler.Xla.PJRT.C.PjrtCApi
import Device
import Tensor

import Utils.Comparison
import Utils.Cases

import Data.SOP
import Hedgehog

import Device

import TestUtils
import Composite.Concurrent
import Utils.TestComparison
import Unit.Model.TestKernel
import Unit.TestDistribution
import Unit.TestTensor
import Unit.TestLiteral
import Unit.TestUtil

public export
0 protocol : Session
protocol = Send [] S32 $ Recv [] S32 $ End

-- TagT1 because we can only use the computation (L IO) once, else we'd use the channel n times
-- L (not L1) because we can use the tensor as much as we want
onHost : Channel Main.protocol -@ TagT1 (L IO) (Tensor [] S32)
onHost ch =
    let x = tensor {dtype = S32} 2 in
    lift1 (send ch x DEVICE_TO_DEVICE) `bind1` \ch =>
      recv ch DEVICE_TO_DEVICE `bind1` \(x # ch) =>
        lift (end ch) `bind` \() =>
          pure x

onDevice : Channel (dual Main.protocol) -@ TagT1 (L IO) ()
onDevice ch = do
    recv ch DEVICE_TO_DEVICE `bind1` \(x # ch) =>
      lift1 (send ch x DEVICE_TO_DEVICE) `bind1` \ch =>
        lift (end ch)

sendRecv : Device -> Device -> Property
sendRecv c0 c1 = fixedProperty $ do
  let MkDevice api client = c0
      devices := do
          -- putStrLn "c0"
          devices <- pjrtClientDevices api client
          descr <- traverse (pjrtDeviceGetDescription api) devices
          debugs <- traverse (pjrtDeviceDescriptionDebugString api) descr
          printLn debugs
          pure devices

      Right [gpu0] = unsafePerformIO (runEitherT devices) | _ => ?notExactlyOneDevice0

      MkDevice api client = c1
      devices := do
          -- putStrLn "c1"
          devices <- pjrtClientDevices api client
          descr <- traverse (pjrtDeviceGetDescription api) devices
          debugs <- traverse (pjrtDeviceDescriptionDebugString api) descr
          printLn debugs
          pure devices

      Right [gpu1] = unsafePerformIO (runEitherT devices) | _ => ?notExactlyOneDevice1

      prog : L IO (Literal [] Int32) = do
        (h # d) <- makeChannel Main.protocol
        eval1nil c0 gpu0 (onDevice d)
        eval1 c1 gpu1 (onHost h)

  unsafePerformIO (LIO.run prog) === 2

group : Device -> Device -> Group
group c0 c1 = MkGroup "Concurrent" $ [
      ("send and recv a simple value, no maths", sendRecv c0 c1)
  ]

run : Device -> Device -> IO ()
run c0 c1 = test [group c0 c1]

partial
main : IO ()
main = do
  Right c0 <- runEitherT (device 0.1) | Left err => die $ show err
  Right c1 <- runEitherT (device 0.1) | Left err => die $ show err
  run c0 c1

-- problem of mhlo marked illegal: can we just stick with mhlo since we're no longer using a mac,
-- that might be enough to make everything work
