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
