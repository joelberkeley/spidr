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
module Compiler.MLIR.IR.Operation

import Compiler.MLIR.IR.Value
import Compiler.MLIR.IR.ValueRange
import Compiler.FFI

public export
data Operation = MkOperation GCAnyPtr

%foreign (libxla "Operation_delete")
prim__deleteOperation : AnyPtr -> PrimIO ()

%foreign (libxla "Operation_erase")
prim__operationErase : GCAnyPtr -> PrimIO ()

export
erase : HasIO io => Operation -> io ()
erase (MkOperation op) = primIO $ prim__operationErase op

%foreign (libxla "Operation_getOpResults")
prim__operationGetOpResults : GCAnyPtr -> PrimIO $ AnyPtr

export
getOpResults : HasIO io => Operation -> io ResultRange
getOpResults (MkOperation op) = do
  res <- primIO $ prim__operationGetOpResults op
  res <- onCollectAny res (primIO . prim__deleteResultRange)
  pure (MkResultRange res)

%foreign (libxla "Operation_getOpResult")
prim__operationGetOpResult : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

export
getOpResult : HasIO io => Operation -> Nat -> io OpResult
getOpResult (MkOperation op) idx = do
  res <- primIO $ prim__operationGetOpResult op (cast idx)
  res <- onCollectAny res (primIO . prim__deleteOpResult)
  pure (MkOpResult res)
