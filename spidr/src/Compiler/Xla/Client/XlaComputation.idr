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
||| For internal spidr use only.
module Compiler.Xla.Client.XlaComputation

import Compiler.FFI

public export
data XlaComputation : Type where
  MkXlaComputation : GCAnyPtr -> XlaComputation

%foreign (libxla "XlaComputation_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__delete

{-
until I work out how to handle memory of an HloModuleProto

export
data HloModuleProto = MkHloModuleProto GCAnyPtr

export
proto : XlaComputation -> HloModuleProto
proto (MkXlaComputation computation) = do

  pure $ MkHloModuleProto proto

-- doesn't belong here, see cpp
export
%foreign (libxla "HloModuleProto_SerializeAsString")
prim__hloModuleProtoSerializeAsString : AnyPtr -> PrimIO String
-}

export
%foreign (libxla "XlaComputation_SerializeAsString")
prim__xlaComputationSerializeAsString : GCAnyPtr -> PrimIO AnyPtr

||| It is up to the caller to deallocate the CharArray.
export
serializeAsString : HasIO io => XlaComputation -> io CharArray
serializeAsString (MkXlaComputation computation) = do
  str <- primIO $ prim__xlaComputationSerializeAsString computation
  data' <- primIO $ prim__stringData str
  let size = prim__stringSize str
  primIO $ prim__stringDelete str
  pure (MkCharArray data' size)
