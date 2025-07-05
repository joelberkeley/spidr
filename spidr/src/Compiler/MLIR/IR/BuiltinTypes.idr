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
module Compiler.MLIR.IR.BuiltinTypes

import Compiler.MLIR.IR.BuiltinTypeInterfaces
import Compiler.MLIR.IR.MLIRContext
import Compiler.MLIR.IR.Operation
import Compiler.MLIR.IR.Types
import Compiler.MLIR.IR.TypeRange
import Compiler.FFI

public export
data FloatType = MkFloatType GCAnyPtr

export
%foreign (libxla "FloatType_delete")
prim__deleteFloatType : AnyPtr -> PrimIO ()

%foreign (libxla "set_array_FloatType")
prim__setArrayFloatType : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

export
Cast FloatType Type_ where
  cast (MkFloatType t) = MkType_ t prim__setArrayFloatType

public export
data FunctionType = MkFunctionType GCAnyPtr

%foreign (libxla "FunctionType_delete")
prim__deleteFunctionType : AnyPtr -> PrimIO ()

%foreign (libxla "FunctionType_get")
prim__functionTypeGet : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

namespace FunctionType
  export
  get : HasIO io => MLIRContext -> TypeRange -> TypeRange -> io FunctionType
  get (MkMLIRContext ctx) (MkTypeRange inputs) (MkTypeRange results) = do
    ftype <- primIO $ prim__functionTypeGet ctx inputs results
    ftype <- onCollectAny ftype (primIO . prim__deleteFunctionType)
    pure (MkFunctionType ftype)

public export
data RankedTensorType = MkRankedTensorType GCAnyPtr

%foreign (libxla "set_array_RankedTensorType")
prim__setArrayRankedTensorType : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

%foreign (libxla "RankedTensorType_delete")
prim__deleteRankedTensorType : AnyPtr -> PrimIO ()

namespace ShapedType
  export
  Cast RankedTensorType ShapedType where
    cast (MkRankedTensorType t) = MkShapedType t

namespace Type_
  export
  Cast RankedTensorType Type_ where
    cast (MkRankedTensorType t) = MkType_ t prim__setArrayRankedTensorType

%foreign (libxla "RankedTensorType_get")
prim__rankedTensorTypeGet : GCPtr Int64 -> Bits64 -> GCAnyPtr -> PrimIO AnyPtr

namespace RankedTensorType
  export
  get : HasIO io => List Nat -> Types.Type_ -> io RankedTensorType
  get shape (MkType_ elementType _) = do
    MkInt64Array arr <- mkInt64Array (map cast shape)
    rtt <- primIO $ prim__rankedTensorTypeGet arr (cast $ length shape) elementType
    rtt <- onCollectAny rtt (primIO . prim__deleteRankedTensorType)
    pure (MkRankedTensorType rtt)
