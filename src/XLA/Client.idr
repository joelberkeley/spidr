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
module XLA.Client

import System.FFI

import XLA.Literal
import XLA.FFI
import XLA.Client.ClientLibrary
import XLA.Client.LocalClient
import XLA.Client.XlaBuilder
import XLA.Client.XlaComputation
import XLA.XlaData

import Types

export
eval : XLAPrimitive dtype => {shape : _} -> RawTensor -> IO (Array shape dtype)
eval (MkRawTensor f) = do
    builder <- prim__mkXlaBuilder ""
    _ <- f builder
    computation <- prim__build builder
    client <- primIO prim__localClientOrDie
    lit <- prim__executeAndTransfer client computation prim__getNullAnyPtr 0
    pure (toArray lit)
