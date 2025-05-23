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

import System.FFI

import public Compiler.Xla.PJRT.C.PjrtCApi
import public Device

%foreign "C:GetPjrtApi,pjrt_plugin_xla_cpu"
prim__getPjrtApi : PrimIO AnyPtr

export
device : Pjrt Device
device = do
  api <- MkPjrtApi <$> primIO prim__getPjrtApi
  pjrtPluginInitialize api
  MkDevice api <$> pjrtClientCreate api
