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
module Compiler.MLIR.IR.BuiltinOps

import Compiler.MLIR.IR.MLIRContext
import Compiler.FFI

public export
data ModuleOp = MkModuleOp GCAnyPtr

export
%foreign (libxla "ModuleOp_delete")
prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "ModuleOp_getContext")
prim__moduleOp : GCAnyPtr -> PrimIO AnyPtr

export
getContext : HasIO io => ModuleOp -> io MLIRContext
getContext (MkModuleOp op) = do
  ctx <- primIO $ prim__moduleOp op
  ctx <- onCollectAny ctx (const $ pure ())  --  I reckon we've already GC'ed this
  pure (MkMLIRContext ctx)
