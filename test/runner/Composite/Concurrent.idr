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

import Control.Monad.State
import Data.Linear.Notation

import Compiler.Xla.XlaData
import Device
import Tensor

import Utils.Comparison
import Utils.Cases

-- tries to use single device
simpleSend : Device => Property
simpleSend = fixedProperty $ do
  let x = tensor {dtype = F64} 2.0
      -- what do we do with this? should this be in IO as there's no value here?
      _ = session $ \s => let s = send s x 1 DEVICE_TO_DEVICE in end s (U ())
      w = session $ \s => MkTagT $ ST $ \env =>
            let (env, x) # s = recv s 1 {shape = [], dtype = F64} DEVICE_TO_DEVICE env
             in U (env, x)
  w ===# pure x

export
group : Device => Group
group = MkGroup "Concurrent" $ [
      ("send/recv a simple value, no operations", simpleSend)
  ]
