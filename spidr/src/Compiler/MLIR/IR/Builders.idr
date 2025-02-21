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

import Compiler.MLIR.IR.MLIRContext
import Compiler.FFI

public export
data OpBuilder = MkOpBuilder GCAnyPtr

%foreign (libxla "OpBuilder_new")
prim__mkOpBuilder : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "OpBuilder_delete")
prim__deleteOpBuilder : AnyPtr -> PrimIO ()

export
mkOpBuilder : HasIO io => MLIRContext -> io OpBuilder
mkOpBuilder (MkMLIRContext ctx) = do
  op <- primIO $ prim__mkOpBuilder ctx
  op <- onCollectAny op (primIO . prim__deleteOpBuilder)
  pure (MkOpBuilder op)

export
getF64Type : HasIO io => OpBuilder -> io Types.Type
