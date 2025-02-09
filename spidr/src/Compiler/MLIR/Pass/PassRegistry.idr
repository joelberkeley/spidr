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
module Compiler.MLIR.Pass.PassRegistry

import Compiler.LLVM.Support.RawOStream
import Compiler.MLIR.Pass.PassManager
import Compiler.FFI

%foreign (libxla "parsePassPipeline")
prim__parsePassPipeline : String -> GCAnyPtr -> GCAnyPtr -> PrimIO Int

export
parsePassPipeline : HasIO io => String -> PassManager -> RawStringOStream -> io Bool
parsePassPipeline pipeline (MkPassManager pm) (MkRawStringOStream errorStream) = do
  ok <- primIO $ prim__parsePassPipeline pipeline pm errorStream
  pure (cIntToBool ok)
