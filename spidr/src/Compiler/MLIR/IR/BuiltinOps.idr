{--
Copyright 2025 Joel Berkeley

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
import Compiler.MLIR.IR.Operation
import Compiler.FFI

public export
data ModuleOp = MkModuleOp GCAnyPtr

export
%foreign (libxla "ModuleOp_delete")
prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "ModuleOp_dump")
prim__moduleOpDump : GCAnyPtr -> PrimIO ()

export
dump : HasIO io => ModuleOp -> io ()
dump (MkModuleOp op) = primIO $ prim__moduleOpDump op

export
%foreign (libxla "ModuleOp_getOperation")
prim__moduleOpGetOperation : GCAnyPtr -> AnyPtr

export
getOperation : HasIO io => ModuleOp -> io Operation
getOperation (MkModuleOp moduleOp) = do
  let op = prim__moduleOpGetOperation moduleOp
  op <- onCollectAny op (const $ pure ())  -- I assume the ModuleOp owns the Operation
  pure (MkOperation op)

export
%foreign (libxla "ModuleOp_push_back")
prim__moduleOpPushBack : GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
pushBack : HasIO io => ModuleOp -> Operation -> io ()
pushBack (MkModuleOp mOp) (MkOperation op) = primIO $ prim__moduleOpPushBack mOp op
