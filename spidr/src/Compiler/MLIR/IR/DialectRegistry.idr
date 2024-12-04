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
module Compiler.MLIR.IR.DialectRegistry

import Compiler.FFI

public export
data DialectRegistry = MkDialectRegistry GCAnyPtr

%foreign (libxla "DialectRegistry_new")
prim__mkDialectRegistry : PrimIO AnyPtr

%foreign (libxla "DialectRegistry_delete")
prim__deleteDialectRegistry : AnyPtr -> PrimIO ()

export
mkDialectRegistry : HasIO io => io DialectRegistry
mkDialectRegistry = do
  registry <- primIO prim__mkDialectRegistry
  registry <- onCollectAny registry (primIO . prim__deleteDialectRegistry)
  pure (MkDialectRegistry registry)
