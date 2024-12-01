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
module Compiler.Xla.Service.HloModuleConfig

import Compiler.FFI
import Compiler.Xla.Shape

public export
data HloModuleConfig = MkHloModuleConfig GCAnyPtr

%foreign (libxla "HloModuleConfig_new")
prim__hloModuleConfig : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "HloModuleConfig_delete")
prim__delete : AnyPtr -> PrimIO ()

export
hloModuleConfig : HasIO io => ProgramShape -> io HloModuleConfig
hloModuleConfig (MkProgramShape pshape) = do
  config <- primIO $ prim__hloModuleConfig pshape
  config <- onCollectAny config (primIO . prim__delete)
  pure (MkHloModuleConfig config)
