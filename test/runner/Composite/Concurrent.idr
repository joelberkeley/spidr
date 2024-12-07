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
  let x = tensor {dtype = S32} 2

      host : Channel Concurrent.protocol -@ L (TagT IO) ()
      host s = do
        s <- send s x HOST_TO_DEVICE
        (env, x) # s <- recv s HOST_TO_DEVICE
        end s

      device : Channel Concurrent.protocol -@ L (TagT IO) ()

      prog : TagT IO (Tensor [] S32) = fork Concurrent.protocol host ?device'

        {-
        the (TagT IO _) $ lift $ MkTagT $ ST $ \env => the (IO _) $ run $ the (L IO _) $ do
        session $
        session $ \s => the (L1 IO ()) $ do
          (env, x) # s <- the (L1 IO _) $ recv s {shape = [], dtype = S32} DEVICE_TO_HOST env
          the (L IO ()) $ pure $ end s (env, x)
          -}
  Tensor.Tag.eval prog === 2

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send/recv a simple value, no operations", simpleSend)
  ]
