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
module Compiler.Xla.Xla.PJRT.C.PJRT_C_API_CPU

import System.FFI

import Compiler.Xla.Prim.Util

-- we're going to need to alias this C function so we can differentiate between
-- versions of GetPjrtApi for various devices
%foreign (libxla "GetPjrtApi")
prim__getPjrtApi : PrimIO AnyPtr

export
getPjrtApi : HasIO io => io PjRtApi
getPjrtApi = MkPJRT_Api <&> primIO prim__getPjrtApi
