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
module Compiler.Xla.Xla.ShapeUtil

import Compiler.Xla.Prim.Xla.ShapeUtil
import Compiler.Xla.Xla.Shape
import Compiler.Xla.Xla.XlaData
import Compiler.Xla.Util
import Types

public export
data ShapeIndex : Type where
  MkShapeIndex : GCAnyPtr -> ShapeIndex

namespace ShapeIndex
  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . prim__shapeIndexDelete

export
allocShapeIndex : HasIO io => io ShapeIndex
allocShapeIndex = do
  ptr <- primIO prim__shapeIndexNew
  ptr <- onCollectAny ptr ShapeIndex.delete
  pure (MkShapeIndex ptr)

export
pushBack : HasIO io => ShapeIndex -> Nat -> io ()
pushBack (MkShapeIndex shapeIndex) value =
  primIO $ prim__shapeIndexPushBack shapeIndex (cast value)

export
pushFront : HasIO io => ShapeIndex -> Nat -> io ()
pushFront (MkShapeIndex shapeIndex) value =
  primIO $ prim__shapeIndexPushFront shapeIndex (cast value)

export
byteSizeOfElements : Xla.Shape -> Bits64
byteSizeOfElements (MkShape shape) = prim__byteSizeOfElements shape

export
mkTupleShape : HasIO io => List Xla.Shape -> io Xla.Shape
mkTupleShape shapes = do
  MkShapeArray shapeArray <- mkShapeArray shapes
  shape <- primIO $ prim__mkTupleShape shapeArray (cast $ length shapes)
  shape <- onCollectAny shape Shape.delete
  pure (MkShape shape)

export
mkShape : (HasIO io, Primitive dtype) => Types.Shape -> io Xla.Shape
mkShape shape = do
  let dtypeEnum = xlaIdentifier {dtype}
  MkIntArray shapeArrayPtr <- mkIntArray shape
  shapePtr <- primIO $ prim__mkShape dtypeEnum shapeArrayPtr (cast $ length shape)
  shapePtr <- onCollectAny shapePtr Shape.delete
  pure (MkShape shapePtr)
