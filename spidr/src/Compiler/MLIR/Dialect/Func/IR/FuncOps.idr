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
module Compiler.MLIR.Dialect.Func.IR.FuncOps

import Compiler.MLIR.IR.Block
import Compiler.MLIR.IR.Builders
import Compiler.MLIR.IR.BuiltinTypes
import Compiler.MLIR.IR.Location
import Compiler.MLIR.IR.Operation
import Compiler.MLIR.IR.TypeRange
import Compiler.MLIR.IR.ValueRange
import Compiler.FFI

%foreign (libxla "CallOp_delete")
prim__deleteCallOp : AnyPtr -> PrimIO ()

public export
data CallOp = MkCallOp GCAnyPtr

%foreign (libxla "OpBuilder_create_CallOp")
prim__opBuilderCreateCallOp :
  GCAnyPtr -> GCAnyPtr -> String -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

namespace OpBuilder
  export
  createCallOp : HasIO io => OpBuilder -> Location -> String -> TypeRange -> ValueRange -> io CallOp
  createCallOp
    (MkOpBuilder builder)
    (MkLocation location)
    name
    (MkTypeRange returnTypes)
    (MkValueRange operands) = do
      op <- primIO $ prim__opBuilderCreateCallOp builder location name returnTypes operands
      op <- onCollectAny op (primIO . prim__deleteCallOp)
      pure (MkCallOp op)

%foreign (libxla "CallOp_getOperation")
prim__callOpGetOperation : GCAnyPtr -> PrimIO AnyPtr

namespace CallOp
  export
  getOperation : HasIO io => CallOp -> io Operation
  getOperation (MkCallOp op) = do
    opr <- primIO $ prim__callOpGetOperation op
    opr <- onCollectAny opr (const $ pure ())
    pure (MkOperation opr)

public export
data FuncOp = MkFuncOp GCAnyPtr

%foreign (libxla "FuncOp_delete")
prim__deleteFuncOp : AnyPtr -> PrimIO ()

%foreign (libxla "FuncOp_create")
prim__funcOpCreate : GCAnyPtr -> String -> GCAnyPtr -> PrimIO AnyPtr

namespace FuncOp
  export
  create : HasIO io => Location -> String -> FunctionType -> io FuncOp
  create (MkLocation location) name (MkFunctionType type) = do
    op <- primIO $ prim__funcOpCreate location name type
    op <- onCollectAny op (primIO . prim__deleteFuncOp)
    pure (MkFuncOp op)

%foreign (libxla "FuncOp_getOperation")
prim__funcOpGetOperation : GCAnyPtr -> PrimIO AnyPtr

namespace FuncOp
  export
  getOperation : HasIO io => FuncOp -> io Operation
  getOperation (MkFuncOp op) = do
    opr <- primIO $ prim__funcOpGetOperation op
    opr <- onCollectAny opr (const $ pure ())
    pure (MkOperation opr)

public export
data ReturnOp = MkReturnOp GCAnyPtr

%foreign (libxla "ReturnOp_delete")
prim__deleteReturnOp : AnyPtr -> PrimIO ()

%foreign (libxla "OpBuilder_create_ReturnOp")
prim__returnOpCreate : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

namespace OpBuilder
  export
  createReturnOp : HasIO io => OpBuilder -> Location -> ResultRange -> io ReturnOp
  createReturnOp (MkOpBuilder builder) (MkLocation location) (MkResultRange results) = do
    op <- primIO $ prim__returnOpCreate builder location results
    op <- onCollectAny op (primIO . prim__deleteReturnOp)
    pure (MkReturnOp op)

%foreign (libxla "FuncOp_addEntryBlock")
prim__funcOpAddEntryBlock : GCAnyPtr -> PrimIO AnyPtr

export
addEntryBlock : HasIO io => FuncOp -> io Block
addEntryBlock (MkFuncOp op) = do
  block <- primIO $ prim__funcOpAddEntryBlock op
  block <- onCollectAny block (primIO . prim__deleteBlock)
  pure (MkBlock block)
