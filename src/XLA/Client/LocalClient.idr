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

%foreign (libxla "LocalClient_TransferToServer")
prim__transferToServerImpl : LocalClient -> GCAnyPtr -> PrimIO AnyPtr

export
prim__transferToServer : LocalClient -> GCAnyPtr -> IO GCAnyPtr
prim__transferToServer client literal = do
  global_data <- primIO (prim__transferToServerImpl client literal)
  onCollectAny global_data free

%foreign (libxla "LocalClient_ExecuteAndTransfer")
prim__executeAndTransferImpl : LocalClient -> GCAnyPtr -> AnyPtr -> Int -> PrimIO AnyPtr

export
prim__executeAndTransfer : LocalClient -> GCAnyPtr -> AnyPtr -> Int -> IO GCAnyPtr
prim__executeAndTransfer client computation arguments arguments_len = do
  literal <- primIO (prim__executeAndTransferImpl client computation arguments arguments_len)
  onCollectAny literal Literal.delete
