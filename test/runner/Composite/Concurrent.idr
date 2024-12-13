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
module Composite.Concurrent

import Control.Linear.LIO
import Control.Monad.State
import Data.Linear.Notation

import Compiler.Xla.XlaData
import Compiler.Xla.PJRT.C.PjrtCApi
import Device
import Tensor

import Utils.Comparison
import Utils.Cases

public export
0 protocol : Session
protocol = SendT [] S32 $ RecvT [] S32 $ EndT

-- tries to use single device
sendRecv : Device => Property
sendRecv @{device} = fixedProperty $ do
  let MkDevice api client = device
      Right [dev] = unsafePerformIO (runEitherT (pjrtClientDevices api client)) | _ => ?fhnewoi

  let onHost : Channel Concurrent.protocol -@ TagT1 (L IO) (Tensor [] S32)
      onHost ch = do
        let x = tensor {dtype = S32} 2
        ch <- lift1 $ send ch x DEVICE_TO_DEVICE
        x # ch <- recv ch DEVICE_TO_DEVICE
        lift (end ch) `bind` \_ => pure x

      onDevice : Channel (dual Concurrent.protocol) -@ TagT1 (L IO) ()
      onDevice ch = do
        x # ch <- recv {shape = [], dtype = S32} ch DEVICE_TO_DEVICE
        ch <- lift1 $ send ch x DEVICE_TO_DEVICE
        lift $ end ch

      prog : TagT1 (L IO) (Tensor [] S32) := do
        (h # d) <- lift1 $ makeChannel Concurrent.protocol
        -- this doesn't look very concurrent
        onDevice d `bind` \_ => onHost h

  (unsafePerformIO $ eval1 device dev prog) === 2

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send and recv a simple value, no maths operations", sendRecv)
  ]
