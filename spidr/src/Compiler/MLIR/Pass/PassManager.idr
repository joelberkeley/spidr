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
module Compiler.MLIR.Pass.PassManager

import Compiler.MLIR.IR.MLIRContext
import Compiler.MLIR.IR.Operation
import Compiler.MLIR.Pass.Pass
import Compiler.FFI

public export
data PassManager = MkPassManager GCAnyPtr

%foreign (libxla "PassManager_new")
prim__mkPassManager : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "PassManager_delete")
prim__delete : AnyPtr -> PrimIO ()

export
mkPassManager : HasIO io => MLIRContext -> io PassManager
mkPassManager (MkMLIRContext ctx) = do
  manager <- primIO $ prim__mkPassManager ctx
  manager <- onCollectAny manager (primIO . PassManager.prim__delete)
  pure (MkPassManager manager)

%foreign (libxla "PassManager_addPass")
prim__passManagerAddPass : GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
addPass : HasIO io => PassManager -> Pass -> io ()
addPass (MkPassManager manager) (MkPass pass) = primIO $ prim__passManagerAddPass manager pass

%foreign (libxla "PassManager_run")
prim__passManagerRun : GCAnyPtr -> GCAnyPtr -> PrimIO Int

export
run : HasIO io => PassManager -> Operation -> io Bool
run (MkPassManager manager) (MkOperation op) = do
  ok <- primIO $ prim__passManagerRun manager op
  pure (cIntToBool ok)
