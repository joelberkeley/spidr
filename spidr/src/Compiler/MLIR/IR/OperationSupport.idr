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
module Compiler.MLIR.Pass.OperationSupport

import Compiler.MLIR.IR.Location
import Compiler.FFI

public export
data OperationState = MkOperationState GCAnyPtr

%foreign (libxla "OperationState_new")
prim__mkOperationState : GCAnyPtr -> String -> PrimIO AnyPtr

%foreign (libxla "OperationState_delete")
prim__delete : AnyPtr -> PrimIO ()

export
mkOperationState : HasIO io => Location -> String -> io OperationState
mkOperationState (MkLocation location) name = do
  opState <- primIO $ prim__mkOperationState location name
  opState <- onCollectAny opState (primIO . OperationState.prim__delete)
  pure (MkOperationState opState)

%foreign (libxla "OperationState_addOperands")
prim__operationStateAddOperands : GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
addOperands : HasIO io => OperationState -> ValueRange -> io ()
addOperands (MkOperationState opState) (MkValueRange valueRange) =
  primIO $ prim__operationStateAddOperands opState valueRange

%foreign (libxla "OperationState_addAttribute")
prim__operationStateAddAttribute : GCAnyPtr -> GCAnyPtr -> PrimIO ()

export
addAttribute : HasIO io => OperationState -> Attribute -> io ()
addAttribute (MkOperationState opState) (MkAttribute attribute) =
  primIO $ prim__operationStateAddAttribute opState attribute
