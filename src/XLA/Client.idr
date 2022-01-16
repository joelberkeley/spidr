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
import public XLA.Client.ClientLibrary as XLA.Client
import public XLA.Client.LocalClient as XLA.Client
import public XLA.Client.XlaBuilder as XLA.Client
import public XLA.Client.XlaComputation as XLA.Client

import Types

export
eval : XLAPrimitive dtype => {shape : _} -> RawTensor -> IO (Array shape dtype)
eval (MkRawTensor f) = do
    builder <- primIO (prim__mkXlaBuilder "")
    _ <- f builder
    let computation = build builder
    delete builder
    client <- primIO prim__localClientOrDie
    lit <- primIO $ prim__executeAndTransfer client computation prim__getNullAnyPtr 0
    lit <- onCollectAny lit delete
    delete computation
    let arr = toArray lit
    pure arr
