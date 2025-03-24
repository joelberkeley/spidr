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
module Compiler.MLIR.IR.Builders

import Compiler.MLIR.IR.Block
import Compiler.MLIR.IR.BuiltinTypes
import Compiler.MLIR.IR.MLIRContext
import Compiler.MLIR.IR.Types
import Compiler.FFI

public export
data OpBuilder = MkOpBuilder GCAnyPtr

%foreign (libxla "OpBuilder_delete")
prim__deleteOpBuilder : AnyPtr -> PrimIO ()

%foreign (libxla "OpBuilder_new")
prim__mkOpBuilder : GCAnyPtr -> PrimIO AnyPtr

export
mkOpBuilder : HasIO io => MLIRContext -> io OpBuilder
mkOpBuilder (MkMLIRContext ctx) = do
  builder <- primIO $ prim__mkOpBuilder ctx
  builder <- onCollectAny builder (primIO . prim__deleteOpBuilder)
  pure (MkOpBuilder builder)

%foreign (libxla "OpBuilder_atBlockEnd")
prim__opBuilderAtBlockEnd : GCAnyPtr -> PrimIO AnyPtr

export
atBlockEnd : HasIO io => Block -> io OpBuilder
atBlockEnd (MkBlock block) = do
  builder <- primIO $ prim__opBuilderAtBlockEnd block
  builder <- onCollectAny builder (primIO . prim__deleteOpBuilder)
  pure (MkOpBuilder builder)

%foreign (libxla "Builder_getF64Type")
prim__builderGetF64Type : GCAnyPtr -> PrimIO AnyPtr

export
getF64Type : HasIO io => OpBuilder -> io FloatType
getF64Type (MkOpBuilder builder) = do
  type <- primIO $ prim__builderGetF64Type builder
  type <- onCollectAny type (primIO . prim__deleteFloatType)
  pure (MkFloatType type)
