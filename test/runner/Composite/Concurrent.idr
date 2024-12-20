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

-- TagT1 because we can only use the computation (L IO) once, else we'd use the channel n times
-- L (not L1) because we can use the tensor as much as we want
onHost : Channel Concurrent.protocol -@ TagT1 (L IO) (Tensor [] S32)
onHost ch =
    let x = tensor {dtype = S32} 2 in
    lift1 (send ch x DEVICE_TO_DEVICE) `bind1` \ch =>
      recv ch DEVICE_TO_DEVICE `bind1` \(x # ch) =>
        lift (end ch) `bind` \() =>
          pure x

onDevice : Channel (dual Concurrent.protocol) -@ TagT1 (L IO) ()
onDevice ch = do
    recv ch DEVICE_TO_DEVICE `bind1` \(x # ch) =>
      lift1 (send ch x DEVICE_TO_DEVICE) `bind1` \ch =>
        lift (end ch)

sendRecv : Device => Property
sendRecv @{device} = fixedProperty $ do
  let MkDevice api client = device
      Right [dev] = unsafePerformIO (runEitherT (pjrtClientDevices api client)) | _ => ?notExactlyOneDevice

      prog : L IO (Literal [] Int32) = do
        (h # d) <- makeChannel Concurrent.protocol
        eval1nil device dev (onDevice d)
        eval1 device dev (onHost h)

  unsafePerformIO (run prog) === 2

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send and recv a simple value, no maths", sendRecv)
  ]
