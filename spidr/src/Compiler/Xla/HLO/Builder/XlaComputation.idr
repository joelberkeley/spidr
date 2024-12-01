{--
Copyright 2022 Joel Berkeley

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
module Compiler.Xla.HLO.Builder.XlaComputation

import Compiler.FFI
import Compiler.Xla.Shape
import Compiler.Xla.Service.HloProto

public export
data XlaComputation : Type where
  MkXlaComputation : GCAnyPtr -> XlaComputation

%foreign (libxla "XlaComputation_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . XlaComputation.prim__delete

%foreign (libxla "XlaComputation_GetProgramShape")
prim__xlaComputationGetProgramShape : GCAnyPtr -> PrimIO AnyPtr

export
getProgramShape : HasIO io => XlaComputation -> io ProgramShape
getProgramShape (MkXlaComputation comp) = do
  pshape <- primIO $ prim__xlaComputationGetProgramShape comp
  pshape <- onCollectAny pshape (primIO . prim__ProgramShape_delete)
  pure (MkProgramShape pshape)

%foreign (libxla "XlaComputation_proto")
prim__xlaComputationProto : GCAnyPtr -> PrimIO AnyPtr

export
proto : HasIO io => XlaComputation -> io HloModuleProto
proto (MkXlaComputation comp) = do
  proto <- primIO $ prim__xlaComputationProto comp
  proto <- onCollectAny proto (primIO . HloProto.prim__delete)
  pure (MkHloModuleProto proto)
