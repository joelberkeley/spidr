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
module Compiler.Xla.PjRt.C.PjRt_C_API

import System.FFI

import Compiler.Xla.Prim.Util

public export
data PjrtApi = MkPjrtApi AnyPtr

export
data PJRT_Program = MkPJRT_Program GCAnyPtr

{-
public export
PJRT_Program : Type
PJRT_Program = Struct "PJRT_Program" [
      ("struct_size", Bits64)
    , ("extension_start", GCAnyPtr)
    , ("code", GCAnyPtr)
    , ("code_size", Bits64)
    , ("format", GCAnyPtr)
    , ("format_size", Bits64)
  ]
-}

-- setField program "code" (believe_me "<prog>")
