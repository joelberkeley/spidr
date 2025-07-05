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
module Compiler.MLIR.IR.ValueRange

import Compiler.MLIR.IR.Value
import Compiler.FFI

public export
data ValueRange = MkValueRange GCAnyPtr

%foreign (libxla "ValueRange_delete")
prim__deleteValueRange : AnyPtr -> PrimIO ()

%foreign (libxla "ValueRange_new")
prim__mkValueRange : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

public export
mkValueRange : HasIO io => List Value -> io ValueRange
mkValueRange values = do
  MkValueArray arr <- mkValueArray values
  vr <- primIO $ prim__mkValueRange arr (cast $ length values)
  vr <- onCollectAny vr (primIO . prim__deleteValueRange)
  pure (MkValueRange vr)

export
%foreign (libxla "ResultRange_delete")
prim__deleteResultRange : AnyPtr -> PrimIO ()

public export
data ResultRange = MkResultRange GCAnyPtr
