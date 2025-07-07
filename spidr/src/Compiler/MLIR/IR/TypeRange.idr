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
module Compiler.MLIR.IR.TypeRange

import Compiler.MLIR.IR.Types
import Compiler.FFI

public export
data TypeRange = MkTypeRange GCAnyPtr

%foreign (libxla "TypeRange_delete")
prim__deleteTypeRange : AnyPtr -> PrimIO ()

%foreign (libxla "TypeRange_new")
prim__mkTypeRange : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

public export
mkTypeRange : HasIO io => List Type_ -> io TypeRange
mkTypeRange types = do
  MkTypeArray arr <- mkTypeArray types
  tr <- primIO $ prim__mkTypeRange arr (cast $ length types)
  tr <- onCollectAny tr (primIO . prim__deleteTypeRange)
  pure (MkTypeRange tr)
