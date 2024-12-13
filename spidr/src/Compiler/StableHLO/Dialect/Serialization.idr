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
||| For internal spidr use only.
module Compiler.StableHLO.Dialect.Serialization

import Compiler.LLVM.Support.RawOStream
import Compiler.MLIR.IR.BuiltinOps
import Compiler.FFI

%foreign (libxla "serializePortableArtifact")
prim__serializePortableArtifact : AnyPtr -> AnyPtr -> GCAnyPtr -> PrimIO Int

export
serializePortableArtifact : HasIO io => ModuleOp -> CppString -> RawStringOStream -> io Bool
serializePortableArtifact (MkModuleOp moduleOp) (MkCppString version) (MkRawStringOStream os) = do
  ok <- primIO $ prim__serializePortableArtifact moduleOp version os
  pure (cIntToBool ok)
