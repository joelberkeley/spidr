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
module Compiler.Enzyme.Enzyme.Enzyme.MLIR.Passes.Passes

import Compiler.MLIR.IR.BuiltinOps
import Compiler.MLIR.IR.DialectRegistry
import Compiler.MLIR.Pass.Pass
import Compiler.FFI

%foreign (libxla "registerenzymePasses")
prim__registerenzymePasses : PrimIO ()

registerenzymePasses : HasIO io => io ()
registerenzymePasses = primIO $ prim__registerenzymePasses

%foreign (libxla "emitEnzymeADOp")
prim__emitEnzymeADOp : GCPtr Int64 -> Bits64 -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO Int

export
enzymeAD : HasIO io => List Nat -> ModuleOp -> DialectRegistry -> PassManager -> io Bool
enzymeAD shape (MkModuleOp op) (MkDialectRegistry registry) (MkPassManager pm) = do
  MkInt64Array shapePtr <- mkInt64Array (map cast shape)  -- int64 is wrong? should be uint64?
  ok <- primIO $ prim__emitEnzymeADOp shapePtr (cast $ length shape) op registry pm
  pure (cIntToBool ok)
