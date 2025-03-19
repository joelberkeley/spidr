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
import Compiler.FFI

public export
data CallOp = MkCallOp GCAnyPtr

%foreign (libxla "CallOp_create")
prim__callOpCreate : GCAnyPtr -> String -> GCAnyPtr -> PrimIO AnyPtr

namespace OpBuilder
  export
  createCallOp : HasIO io => OpBuilder -> Location -> String -> TypeRange -> io CallOp

public export
data FuncOp = MkFuncOp GCAnyPtr

export
Cast FuncOp Operation where
  cast (MkFuncOp op) = MkOperation op

%foreign (libxla "FuncOp_create")
prim__funcOpCreate : GCAnyPtr -> String -> GCAnyPtr -> PrimIO AnyPtr

namespace FuncOp
  export
  create : HasIO io => Location -> String -> FunctionType -> io FuncOp

public export
data ReturnOp = MkReturnOp GCAnyPtr

%foreign (libxla "ReturnOp_create")
prim__returnOpCreate : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> Bits64 -> PrimIO AnyPtr

namespace OpBuilder
  export
  createReturnOp : HasIO io => OpBuilder -> Location -> List Operation -> io ReturnOp

%foreign (libxla "FuncOp_addEntryBlock")
prim__funcOpAddEntryBlock : GCAnyPtr -> PrimIO AnyPtr

export
addEntryBlock : HasIO io => FuncOp -> io Block
addEntryBlock (MkFuncOp op) = do
  block <- primIO $ prim__funcOpAddEntryBlock op
  block <- onCollectAny block (primIO . prim__deleteBlock)
  pure (MkBlock block)
