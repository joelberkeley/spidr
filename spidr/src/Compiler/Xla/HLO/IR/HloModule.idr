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
module Compiler.Xla.HLO.IR.HloModule

import Compiler.FFI

public export
data HloModule = MkHloModule GCAnyPtr

%foreign (libxla "HloModule_delete")
prim__hloModuleDelete : AnyPtr -> PrimIO ()

%foreign (libxla "HloModule_CreateFromProto")
prim__hloModuleCreateFromProto : AnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
createFromProto : HasIO io => HloModuleProto -> HloModuleConfig -> io HloModule
createFromProto (MkHloModuleProto proto) (MkHloModuleConfig config) = do
  module <- primIO $ prim__hloModuleCreateFromProto proto config
  module <- onCollectAny (primIO . prim__hloModuleDelete) module
  pure (MkHloModule module)
