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
module Device

import Control.Monad.Either

import Compiler.Xla.PJRT.C.PJRT_C_API
import Compiler.Xla.PJRT.C.PJRT_C_API_CPU

import Types

-- it's possible a device will have different clients. atm
-- i'm going to say they're the same thing, and all clients
-- are configured the same. We can separate them when we
-- know why we'd do that
public export
data Device = MkDevice PjrtApi PjrtClient

export
cpu : ErrIO PjrtError Device
cpu = do
  api <- getPjrtApi
  client <- pjrtClientCreate api
  pure (MkDevice api client) 
