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
module PjrtPluginXlaCpu

import System.FFI

import public Compiler.Xla.PJRT.C.PjrtCApi
import public Device

%foreign "C:GetPjrtApi,pjrt_plugin_xla_cpu"
prim__getPjrtApi : PrimIO AnyPtr

export
device : Pjrt Device
device = do
  api <- MkPjrtApi <$> primIO prim__getPjrtApi
  pjrtPluginInitialize api
  MkDevice api <$> pjrtClientCreate api
