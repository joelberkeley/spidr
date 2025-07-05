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
module Compiler.Stablehlo.Dialect.StablehloOps

import Compiler.MLIR.IR.BuiltinAttributes
import Compiler.MLIR.IR.Builders
import Compiler.MLIR.IR.Location
import Compiler.MLIR.IR.Operation
import Compiler.FFI

%foreign (libxla "ConstantOp_delete")
prim__deleteConstantOp : AnyPtr -> PrimIO ()

public export
data ConstantOp = MkConstantOp GCAnyPtr

%foreign (libxla "OpBuilder_create_ConstantOp")
prim__opBuilderCreateConstantOp : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

namespace OpBuilder
  export
  createConstantOp : HasIO io => OpBuilder -> Location -> DenseElementsAttr -> io ConstantOp
  createConstantOp (MkOpBuilder builder) (MkLocation location) (MkDenseElementsAttr attr) = do
    op <- primIO $ prim__opBuilderCreateConstantOp builder location attr
    op <- onCollectAny op (primIO . prim__deleteConstantOp)
    pure (MkConstantOp op)

%foreign (libxla "ConstantOp_getOperation")
prim__constantOpGetOperation : GCAnyPtr -> PrimIO AnyPtr

export
getOperation : HasIO io => ConstantOp -> io Operation
getOperation (MkConstantOp op) = do
  opr <- primIO $ prim__constantOpGetOperation op
  opr <- onCollectAny opr (const $ pure ())
  pure (MkOperation opr)
