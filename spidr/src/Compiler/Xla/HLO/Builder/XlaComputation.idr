{--
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
||| For internal spidr use only.
module Compiler.Xla.HLO.Builder.XlaComputation

import Compiler.FFI
import Compiler.Xla.Shape
import Compiler.Xla.Service.HloProto

public export
data XlaComputation : Type where
  MkXlaComputation : GCAnyPtr -> XlaComputation

%foreign (libxla "XlaComputation_new")
prim__mkXlaComputation : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "XlaComputation_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . XlaComputation.prim__delete

export
mkXlaComputation : HasIO io => HloModuleProto -> io XlaComputation
mkXlaComputation (MkHloModuleProto proto) = do
  comp <- primIO $ prim__mkXlaComputation proto
  comp <- onCollectAny comp XlaComputation.delete
  pure (MkXlaComputation comp)

%foreign (libxla "XlaComputation_proto")
prim__xlaComputationProto : GCAnyPtr -> PrimIO AnyPtr

export
proto : HasIO io => XlaComputation -> io HloModuleProto
proto (MkXlaComputation comp) = do
  proto <- primIO $ prim__xlaComputationProto comp
  proto <- onCollectAny proto (primIO . HloProto.prim__delete)
  pure (MkHloModuleProto proto)

%foreign (libxla "XlaComputation_SerializeAsString")
prim__xlaComputationSerializeAsString : GCAnyPtr -> PrimIO AnyPtr

export
serializeAsString : HasIO io => XlaComputation -> io CharArray
serializeAsString (MkXlaComputation computation) = do
  str <- primIO $ prim__xlaComputationSerializeAsString computation
  stringToCharArray (MkCppString str)
