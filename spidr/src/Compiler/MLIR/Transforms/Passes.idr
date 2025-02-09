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
module Compiler.MLIR.Transforms.Passes

import Compiler.MLIR.Pass.PassManager
import Compiler.FFI

%foreign (libxla "PassManager_addPass_CanonicalizerPass")
prim__passManagerAddPassCanonicalizerPass : GCAnyPtr -> PrimIO ()

export
addCanonicalizerPass : HasIO io => PassManager -> io ()
addCanonicalizerPass (MkPassManager pm) = primIO $ prim__passManagerAddPassCanonicalizerPass pm

%foreign (libxla "PassManager_addPass_CSEPass")
prim__passManagerAddPassCSEPass : GCAnyPtr -> PrimIO ()

export
addCSEPass : HasIO io => PassManager -> io ()
addCSEPass (MkPassManager pm) = primIO $ prim__passManagerAddPassCSEPass pm
