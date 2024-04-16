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
module Compiler.TensorFlow.Compiler.Xla.Client.ClientLibrary

import System.FFI

import Compiler.FFI
import Compiler.TensorFlow.Compiler.Xla.Client.LocalClient
import Compiler.TensorFlow.StreamExecutor.Platform

%foreign (libxla "ClientLibrary_GetOrCreateLocalClient")
prim__getOrCreateLocalClient : AnyPtr -> AnyPtr -> Int -> PrimIO AnyPtr

export
getOrCreateLocalClient : HasIO io => Platform -> io LocalClient
getOrCreateLocalClient (MkPlatform platform) = do
  client <- primIO $ prim__getOrCreateLocalClient platform prim__getNullAnyPtr 0
  pure (MkLocalClient client)
