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
module Compiler.Xla.Xla.Client.XlaComputation

import Compiler.Xla.Prim.Xla.Client.XlaComputation

public export
data XlaComputation : Type where
  MkXlaComputation : GCAnyPtr -> XlaComputation

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
-}

export
serializeAsString : HasIO io => XlaComputation -> io String
serializeAsString (MkXlaComputation computation) =
  primIO $ prim__xlaComputationSerializeAsString computation
