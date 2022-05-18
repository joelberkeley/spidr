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
module Compiler.TensorFlow.Compiler.XLA.ShapeUtil

import System.FFI

import Compiler.FFI
import Compiler.TensorFlow.Compiler.XLA.Shape
import Compiler.TensorFlow.Compiler.XLA.XlaData
import Types

%foreign (libxla "MakeShape")
prim__mkShape : Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
mkShape : HasIO io => Primitive dtype => Shape -> io GCAnyPtr
mkShape shape = do
  let dtypeEnum = xlaIdentifier {dtype}
  shapePtr <- primIO $ prim__mkShape dtypeEnum !(mkIntArray shape) (cast (length shape))
  onCollectAny shapePtr Shape.delete
