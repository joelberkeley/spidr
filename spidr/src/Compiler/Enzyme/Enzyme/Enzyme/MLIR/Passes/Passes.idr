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

%foreign (libxla "createDifferentiatePass")
prim__createDifferentiatePass : PrimIO AnyPtr

export
createDifferentiatePass : HasIO io => io Pass
createDifferentiatePass = do
  pass <- primIO prim__createDifferentiatePass
  pass <- onCollectAny pass (primIO . Pass.prim__delete)
  pure (MkPass pass)

%foreign (libxla "emitEnzymeADOp")
prim__emitEnzymeADOp : GCAnyPtr -> PrimIO AnyPtr

export
emitEnzymeADOp : HasIO io => ModuleOp -> io ModuleOp
emitEnzymeADOp (MkModuleOp op) = do
  op <- primIO $ prim__emitEnzymeADOp op
  op <- onCollectAny op (primIO . BuiltinOps.prim__delete)
  pure (MkModuleOp op)
