{--
Copyright (C) 2024  Joel Berkeley

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
module Compiler.Xla.PJRT.PjrtExecutable

import Compiler.Xla.Client.ExecutableBuildOptions

import Compiler.FFI

export
data CompileOptions = MkCompileOptions GCAnyPtr

%foreign (libxla "CompileOptions_new")
prim__mkCompileOptions : AnyPtr -> PrimIO AnyPtr

export
mkCompileOptions : HasIO io => ExecutableBuildOptions -> io CompileOptions
mkCompileOptions (MkExecutableBuildOptions executableBuildOptions) = do
  options <- primIO $ prim__mkCompileOptions executableBuildOptions
  options <- onCollectAny options free
  pure (MkCompileOptions options)

%foreign (libxla "CompileOptions_SerializeAsString")
prim__compileOptionsSerializeAsString : GCAnyPtr -> PrimIO AnyPtr

||| It is up to the caller to deallocate the CharArray.
export
serializeAsString : HasIO io => CompileOptions -> io CharArray
serializeAsString (MkCompileOptions options) = do
  str <- primIO $ prim__compileOptionsSerializeAsString options
  data' <- primIO $ prim__stringData str
  let size = prim__stringSize str
  primIO $ prim__stringDelete str
  pure (MkCharArray data' size)
