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
module Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil

import System.FFI

import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.Util
import Types
import Util

export
mkShape : (HasIO io, Primitive dtype) => Types.Shape -> io Xla.Shape
mkShape shape = do
  let dtypeEnum = xlaIdentifier {dtype}
  MkIntArray shapeArrayPtr <- mkIntArray shape
  shapePtr <- primIO $ prim__mkShape dtypeEnum shapeArrayPtr (cast $ length shape)
  shapePtr <- onCollectAny shapePtr Shape.delete
  pure (MkShape shapePtr)

export
mkTupleShape : HasIO io => List Xla.Shape -> io Xla.Shape
mkTupleShape shapes = do
  shapeArray <- malloc (cast (length shapes) * sizeofShape)
  traverse_ (\(idx, MkShape shape) => do
      primIO $ prim__setArrayShape shapeArray (cast idx) shape
    ) (enumerate shapes)
  shapeArray <- onCollectAny shapeArray free
  shapePtr <- primIO $ prim__mkTupleShape shapeArray (cast $ length shapes)
  shapePtr <- onCollectAny shapePtr Shape.delete
  pure (MkShape shapePtr)
