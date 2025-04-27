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
module Compiler.MLIR.IR.SymbolTable

import Compiler.MLIR.IR.Operation
import Compiler.FFI

%foreign (libxla "SymbolTable_lookupSymbolIn")
prim__symbolTableLookupSymbolIn : GCAnyPtr -> String -> PrimIO AnyPtr

export
lookupSymbolIn : HasIO io => Operation -> String -> io Operation
lookupSymbolIn (MkOperation op) symbol = do
  op <- primIO $ prim__symbolTableLookupSymbolIn op symbol
  op <- onCollectAny op (const $ pure ())
  pure (MkOperation op)
