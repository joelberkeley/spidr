{--
Copyright (C) 2025  Joel Berkeley

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
module Compiler.Xla.Client.ExecutableBuildOptions

import Compiler.FFI

public export
data ExecutableBuildOptions = MkExecutableBuildOptions AnyPtr

%foreign (libxla "ExecutableBuildOptions_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : HasIO io => ExecutableBuildOptions -> io ()
delete (MkExecutableBuildOptions opts) = primIO $ prim__delete opts

%foreign (libxla "ExecutableBuildOptions_new")
prim__mkExecutableBuildOptions : PrimIO AnyPtr

export
mkExecutableBuildOptions : HasIO io => io ExecutableBuildOptions
mkExecutableBuildOptions = MkExecutableBuildOptions <$> primIO prim__mkExecutableBuildOptions
