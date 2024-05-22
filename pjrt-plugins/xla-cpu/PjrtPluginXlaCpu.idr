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
module PjrtPluginXlaCpu

import Control.Monad.Either
import System.FFI

import Device
import Compiler.Xla.PJRT.C.PJRT_C_API

%foreign "C:GetPjrtApi,pjrt_plugin_xla_cpu"
prim__getPjrtApi : PrimIO AnyPtr

export
device : PjrtFFI Device
device = do
  api <- MkPjrtApi <$> primIO prim__getPjrtApi
  MkDevice api <$> pjrtClientCreate api
