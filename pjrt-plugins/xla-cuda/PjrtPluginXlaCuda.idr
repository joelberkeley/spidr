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

import Data.SortedMap
import System.FFI

import public Compiler.Xla.PJRT.C.PjrtCApi
import public Device

%foreign "C:GetPjrtApi,pjrt_plugin_xla_cuda"
prim__getPjrtApi : PrimIO AnyPtr

export
device :
  (memoryFraction : Double) ->
  {auto 0 memoryFractionPositive : (0.0 < memoryFraction) === True} ->
  {auto 0 memoryFractionLtOne : (memoryFraction <= 1.0) === True} ->
  {-(preallocate : Bool) ->
  (collectiveMemorySize : Int64) ->
  (visibleDevices : List Int64) ->-}
  Pjrt Device
device memoryFraction = do
  api <- primIO prim__getPjrtApi
  let api = MkPjrtApi api
  MkDevice api <$> pjrtClientCreate api (fromList
                                            [ ("memory_fraction", PjrtValueFloat memoryFraction)
                                            {-, ("preallocate", preallocate)
                                            , ("collective_memory_size", collectiveMemorySize)
                                            , ("visible_devices", visibleDevices)-}
                                            ])
