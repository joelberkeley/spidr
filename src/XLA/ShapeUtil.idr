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
module XLA.ShapeUtil

import System.FFI

import XLA.FFI
import XLA.Shape
import XLA.XlaData
import Types

export
%foreign (libxla "MakeShape")
prim__mkShape : Int -> Ptr Int -> Int -> PrimIO AnyPtr

export
mkShape : XLAPrimitive dtype => Shape -> IO GCAnyPtr
mkShape {dtype} shape = do
  c_shape <- mkIntArray shape
  let dtype_enum = cast $ primitiveType {dtype=dtype}
  xla_shape <- primIO $ prim__mkShape dtype_enum c_shape (cast $ length shape)
  free c_shape
  onCollectAny xla_shape delete
