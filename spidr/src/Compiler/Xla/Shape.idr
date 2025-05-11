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
sizeOfShape : Bits64

%foreign (libxla "delete_array_Shape")
prim__deleteArrayShape : AnyPtr -> PrimIO ()

%foreign (libxla "new_array_Shape")
prim__newArrayShape : Bits64 -> PrimIO AnyPtr

%foreign (libxla "set_array_Shape")
prim__setArrayShape : AnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

--%foreign (libxla "shapearray")
--prim__shapearray : GCAnyPtr -> PrimIO AnyPtr

public export
data ShapeArray = MkShapeArray GCAnyPtr

export
mkShapeArray : HasIO io => List Shape -> io ShapeArray
mkShapeArray shapes = do
  arr <- primIO $ prim__newArrayShape $ cast (length shapes)
  traverse_ (\(idx, MkShape shape) => primIO $ prim__setArrayShape arr (cast idx) shape) (enumerate shapes)
  arr <- onCollectAny arr (primIO . prim__deleteArrayShape)
  pure (MkShapeArray arr)
