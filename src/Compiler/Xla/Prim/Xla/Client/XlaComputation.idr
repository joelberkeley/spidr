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
module Compiler.Xla.Prim.Xla.Client.XlaComputation

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "XlaComputation_delete")
prim__delete : AnyPtr -> PrimIO ()

{-
until I work out how to memory handle an HloModuleProto

-- doesn't belong here, see cpp
export
%foreign (libxla "HloModuleProto_SerializeAsString")
prim__hloModuleProtoSerializeAsString : AnyPtr -> PrimIO String
-}

export
%foreign (libxla "XlaComputation_SerializeAsString")
prim__xlaComputationSerializeAsString : GCAnyPtr -> PrimIO String
