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
module Compiler.Enzyme.MLIR.Dialect.Dialect

import Compiler.MLIR.IR.DialectRegistry
import Compiler.MLIR.IR.MLIRContext
import Compiler.FFI

%foreign (libxla "DialectRegistry_insert_EnzymeDialect")
prim__dialectRegistryInsertEnzymeDialect : GCAnyPtr -> PrimIO ()

export
insertEnzymeDialect : HasIO io => DialectRegistry -> io ()
insertEnzymeDialect (MkDialectRegistry reg) = primIO $ prim__dialectRegistryInsertEnzymeDialect reg

%foreign (libxla "MLIRContext_loadDialect_EnzymeDialect")
prim__loadDialectEnzymeDialect : GCAnyPtr -> PrimIO ()

export
loadDialectEnzymeDialect : HasIO io => MLIRContext -> io ()
loadDialectEnzymeDialect (MkMLIRContext ctx) = primIO $ prim__loadDialectEnzymeDialect ctx
