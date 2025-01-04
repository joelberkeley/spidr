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
module Compiler.MLIR.IR.MLIRContext

import Compiler.MLIR.IR.DialectRegistry
import Compiler.FFI

public export
data MLIRContext = MkMLIRContext GCAnyPtr

%foreign (libxla "MLIRContext_new")
prim__mkMLIRContext : PrimIO AnyPtr

%foreign (libxla "MLIRContext_delete")
prim__deleteMLIRContext : AnyPtr -> PrimIO ()

export
mkMLIRContext : HasIO io => io MLIRContext
mkMLIRContext = do
  ctx <- primIO prim__mkMLIRContext
  ctx <- onCollectAny ctx (primIO . prim__deleteMLIRContext)
  pure (MkMLIRContext ctx)

%foreign (libxla "MLIRContext_getDialectRegistry")
prim__getDialectRegistry : GCAnyPtr -> PrimIO AnyPtr

export
getDialectRegistry : HasIO io => MLIRContext -> io DialectRegistry
getDialectRegistry (MkMLIRContext ctx) = do
  registry <- primIO $ prim__getDialectRegistry ctx
  registry <- onCollectAny registry (const $ pure ())  -- correct?
  pure (MkDialectRegistry registry)

%foreign (libxla "MLIRContext_appendDialectRegistry")
prim__appendDialectRegistry : GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
appendDialectRegistry : HasIO io => MLIRContext -> DialectRegistry -> io ()
appendDialectRegistry (MkMLIRContext ctx) (MkDialectRegistry registry) =
  primIO $ prim__appendDialectRegistry ctx registry
