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
import Device
import Tensor

import Utils.Comparison
import Utils.Cases

public export
0 protocol : Session
protocol = SendT [] S32 $ RecvT [] S32 $ EndT

-- tries to use single device
simpleSend : Device => Property
simpleSend = fixedProperty $ do
  let host : Channel Concurrent.protocol -@ TagT1 IO (Tensor [] S32)
      host ch = do
        let x = tensor {dtype = S32} 2
        ch <- send ch x HOST_TO_DEVICE
        x # ch <- recv ch HOST_TO_DEVICE
        end ch  -- i think we need an equivalent to `liftIO1 : (1 _ : IO a) -> io a`, maybe also need `run`
        pure x

      device : Channel Concurrent.protocol -@ TagT1 IO ()
      device ch = do
        x # ch <- recv ch DEVICE_TO_HOST
        ch <- send ch x DEVICE_TO_HOST
        end ch

      prog : TagT IO (Tensor [] S32) = do
        (h # d) <- makeChannel Concurrent.protocol
        -- this doesn't look very concurrent
        host h
        device d

  Tensor.Tag.eval prog === 2

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send/recv a simple value, no operations", simpleSend)
  ]
