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
module Compiler.MLIR.IR.Block

import Compiler.MLIR.IR.Value
import Compiler.FFI

public export
data Block = MkBlock GCAnyPtr

export
%foreign (libxla "Block_delete")
prim__deleteBlock : AnyPtr -> PrimIO ()

export
%foreign (libxla "Block_getArgument")
prim__blockGetArgument : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

export
getArgument : HasIO io => Block -> Nat -> io BlockArgument
getArgument (MkBlock block) i = do
  arg <- primIO $ prim__blockGetArgument block (cast i)
  arg <- onCollectAny arg (primIO . prim__deleteBlockArgument)
  pure (MkBlockArgument arg)
