{--
Copyright 2024 Joel Berkeley

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
module Compiler.StableHLO.Dialect.Register

import Compiler.MLIR.IR.DialectRegistry
import Compiler.FFI

%foreign (libxla "registerAllDialects")
prim__registerAllDialects : GCAnyPtr -> PrimIO ()

export
registerAllDialects : HasIO io => DialectRegistry -> io ()
registerAllDialects (MkDialectRegistry reg) = primIO $ prim__registerAllDialects reg
