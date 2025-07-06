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
module Compiler.Xla.Shape

import Util
import Compiler.FFI

namespace Xla
  public export
  data Shape : Type where
    MkShape : GCAnyPtr -> Shape

%foreign (libxla "Shape_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__delete

%foreign (libxla "sizeof_Shape")
sizeOfShape : Int

%foreign (libxla "set_array_Shape")
prim__setArrayShape : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

public export
data ShapeArray = MkShapeArray GCAnyPtr

export
mkShapeArray : HasIO io => List Shape -> io ShapeArray
mkShapeArray shapes = do
  arr <- malloc (cast (length shapes) * sizeOfShape)
  traverse_ (\(idx, MkShape shape) =>
    primIO $ prim__setArrayShape arr (cast idx) shape) (enumerate (fromList shapes))
  arr <- onCollectAny arr free
  pure (MkShapeArray arr)
