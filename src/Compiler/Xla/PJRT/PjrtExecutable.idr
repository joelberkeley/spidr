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
module Compiler.Xla.PJRT.PjrtExecutable

import Compiler.FFI

export
data CompileOptions = MkCompileOptions GCAnyPtr

%foreign (libxla "CompileOptions_new")
prim__mkCompileOptions : PrimIO AnyPtr

export
mkCompileOptions : HasIO io => io CompileOptions
mkCompileOptions = do
  options <- primIO prim__mkCompileOptions
  -- is `free` sufficient? does CompileOptions own any of its members?
  options <- onCollectAny options free
  pure (MkCompileOptions options)

%foreign (libxla "CompileOptions_SerializeAsString")
prim__compileOptionsSerializeAsString : GCAnyPtr -> PrimIO String

export
serializeAsString : HasIO io => CompileOptions -> io String
serializeAsString (MkCompileOptions options) =
  primIO $ prim__compileOptionsSerializeAsString options