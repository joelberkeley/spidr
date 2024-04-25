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
module PjrtCpuPlugin

import Data.SortedMap
import System.FFI

import Compiler.Xla.PJRT.C.PJRT_C_API

%foreign "C:GetPjrtApi,pjrt_c_api_cpu_plugin"
prim__getPjrtApi : PrimIO AnyPtr

export
getPjrtApi : HasIO io => io PjrtApi
getPjrtApi = MkPjrtApi <$> primIO prim__getPjrtApi

export
createOptions : SortedMap String PjrtValue
createOptions = empty
