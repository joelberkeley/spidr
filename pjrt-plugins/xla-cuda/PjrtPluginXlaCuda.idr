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
module PjrtPluginXlaCuda

import Control.Monad.Either
import Data.SortedMap
import System.FFI

import Device
import Compiler.Xla.PJRT.C.PJRT_C_API
import Types

%foreign "C:GetPjrtApi,pjrt_plugin_xla_cuda"
prim__getPjrtApi : PrimIO AnyPtr

clientCreateOptions : SortedMap String PjrtValue
clientCreateOptions = fromList [
      ("platform_name", PjrtString "cuda")
    , ("allocator", PjrtString "default")
    , ("visible_devices", PjrtInt64Array [0])
  ]

export
device : EitherT PjrtError IO Device
device = do
  api <- MkPjrtApi <$> primIO prim__getPjrtApi
  client <- pjrtClientCreate api clientCreateOptions
  pure $ MkDevice api client