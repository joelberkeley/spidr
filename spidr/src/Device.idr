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

import public Data.SortedMap
import Control.Monad.Either

-- bad import?
import Compiler.Xla.PJRT.C.PJRT_C_API

import Types

public export
data Device = MkDevice PjrtApi PjrtClient

||| Create a PJRT "device". These are required to run spidr graphs, and are provided by your Idris
||| PJRT plugin.
|||
||| @api The core API for the PJRT plugin. This is an Idris reference to a C `PJRT_Api`.
||| @clientCreateOptions Configurations options to a create a C `PJRT_Client`. These are passed as
|||   `.create_options` in the `PJRT_Client_Create_Args` struct.
export
device : (api : PjrtApi) -> (clientCreateOptions : SortedMap String PjrtValue) -> ErrIO PjrtError Device
device api clientCreateOptions = MkDevice api <$> pjrtClientCreate api clientCreateOptions
