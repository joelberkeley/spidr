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

import Compiler.FFI

public export
data ConstantOp = MkConstantOp GCAnyPtr

%foreign (libxla "OpBuilder_create_ConstantOp")
prim__opBuilderCreateConstantOp : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO ()

namespace OpBuilder
  export
  createConstantOp : HasIO io => OpBuilder -> Location -> DenseElementsAttr -> io ConstantOp
  createConstantOp (MkOpBuilder builder) (MkLocation location) (MkDenseElementsAttr attr) = do
    op <- primIO $ prim__opBuilderCreateConstantOp builder location attr
    op <- onCollectAny (primIO . prim__delete) op
    pure (MkConstantOp op)
