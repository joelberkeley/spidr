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
module Compiler.EnzymeJAX.Src.EnzymeAD.JAX.Passes.Passes

import Compiler.MLIR.IR.BuiltinOps
import Compiler.MLIR.IR.MLIRContext
import Compiler.MLIR.Pass.PassManager
import Compiler.FFI

%foreign (libxla "PassManager_addPass_ArithRaisingPass")
prim__passManagerAddPassArithRaisingPass : GCAnyPtr -> PrimIO ()

export
addArithRaisingPass : HasIO io => PassManager -> io ()
addArithRaisingPass (MkPassManager pm) = primIO $ prim__passManagerAddPassArithRaisingPass pm

%foreign (libxla "emitEnzymeADOp")
prim__emitEnzymeADOp : GCPtr Int64 -> Bits64 -> GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
enzymeAD : HasIO io => List Nat -> ModuleOp -> MLIRContext -> io ()
enzymeAD shape (MkModuleOp op) (MkMLIRContext ctx) = do
  MkInt64Array shapePtr <- mkInt64Array (map cast shape)  -- int64 is wrong? should be uint64?
  primIO $ prim__emitEnzymeADOp shapePtr (cast $ length shape) op ctx
