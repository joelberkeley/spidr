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
module Compiler.MLIR.IR.Block

import Compiler.FFI

public export
data Block = MkBlock GCAnyPtr

%foreign (libxla "Block_new")
prim__mkBlock : PrimIO AnyPtr

%foreign (libxla "Block_delete")
prim__deleteBlock : AnyPtr -> PrimIO ()

export
mkBlock : HasIO io => io Block
mkBlock = do
  block <- primIO prim__mkBlock
  block <- onCollectAny block (primIO . prim__deleteBlock)
  pure (MkBlock block)
