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
