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
protocol = Send [] S32 $ Recv [] S32 $ End

-- TagT1 because we can only use the computation (L IO) once, else we'd use the channel n times
-- L (not L1) because we can use the tensor as much as we want
dev0 : Channel Concurrent.protocol -@ TagT1 (L IO) ()
dev0 ch =
  let x = tensor {dtype = S32} 2 in
  lift1 (send ch x) `bind1` \ch =>
    lift (end ch)

dev1 : Channel (dual Concurrent.protocol) -@ TagT1 (L IO) (Tensor [] S32)
dev1 ch = do
  recv ch `bind1` \(x # ch) =>
    lift (end ch) `bind` \() =>
      pure x

sendRecv : Device => Property
sendRecv @{device} = fixedProperty $ do
  let MkDevice api client = device
      devices := do
          devices <- pjrtClientDevices api client
          descr <- traverse (pjrtDeviceGetDescription api) devices
          debugs <- traverse (pjrtDeviceDescriptionDebugString api) descr
          printLn debugs
          pure devices

      Right [gpu0, gpu1] = unsafePerformIO (runEitherT devices) | _ => ?notExactlyTwoDevices

      prog : L IO (Literal [] Int32) = do
        (h # d) <- makeChannel Concurrent.protocol
        -- this might actually work because we don't await any buffers for eval1nil, so we only
        -- wait on the second call. It's obviously super-hacky but meh for now
        eval1nil device gpu0 (onDevice d)
        eval1 device gpu1 (onHost h)

  unsafePerformIO (run prog) === 2

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send and recv a simple value, no maths", sendRecv)
  ]
