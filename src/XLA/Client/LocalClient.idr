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
module XLA.Client.LocalClient

import System.FFI

import XLA.Client.XlaComputation
import XLA.FFI
import XLA.Literal

public export
LocalClient : Type
LocalClient = Struct "LocalClient" []

export
%foreign (libxla "LocalClient_TransferToServer")
prim__transferToServer : LocalClient -> Literal -> PrimIO AnyPtr

export
%foreign (libxla "LocalClient_ExecuteAndTransfer")
prim__executeAndTransfer : LocalClient -> XlaComputation -> AnyPtr -> Int -> PrimIO Literal

export
%foreign (libxla "LocalClient_ExecuteAndTransfer_parameter")
prim__executeAndTransferParameter : LocalClient -> XlaComputation -> Literal -> PrimIO Literal
