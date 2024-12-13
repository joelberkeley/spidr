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
module Compiler.Xla.HLO.Translate.StableHLO

import Compiler.FFI
import Compiler.MLIR.IR.BuiltinOps
import Compiler.MLIR.IR.MLIRContext
import Compiler.Xla.Service.HloProto

%foreign (libxla "ConvertHloToStablehlo")
prim__convertHloToStablehlo : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
convertHloToStablehlo : HasIO io => MLIRContext -> HloModuleProto -> io ModuleOp
convertHloToStablehlo (MkMLIRContext ctx) (MkHloModuleProto proto) = do
  moduleOp <- primIO $ prim__convertHloToStablehlo ctx proto
  moduleOp <- onCollectAny moduleOp (primIO . BuiltinOps.prim__delete)
  pure (MkModuleOp moduleOp)
