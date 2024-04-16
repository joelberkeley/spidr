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
module Compiler.Xla.ShapeUtil

import Compiler.FFI
import Compiler.Xla.Shape
import Compiler.Xla.XlaData
import Types

public export
data ShapeIndex : Type where
  MkShapeIndex : GCAnyPtr -> ShapeIndex

%foreign (libxla "ShapeIndex_delete")
prim__shapeIndexDelete : AnyPtr -> PrimIO ()

namespace ShapeIndex
  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . prim__shapeIndexDelete

%foreign (libxla "ShapeIndex_new")
prim__shapeIndexNew : PrimIO AnyPtr

export
allocShapeIndex : HasIO io => io ShapeIndex
allocShapeIndex = do
  ptr <- primIO prim__shapeIndexNew
  ptr <- onCollectAny ptr ShapeIndex.delete
  pure (MkShapeIndex ptr)

%foreign (libxla "ShapeIndex_push_back")
prim__shapeIndexPushBack : GCAnyPtr -> Int -> PrimIO ()

export
pushBack : HasIO io => ShapeIndex -> Nat -> io ()
pushBack (MkShapeIndex shapeIndex) value =
  primIO $ prim__shapeIndexPushBack shapeIndex (cast value)

%foreign (libxla "ShapeIndex_push_front")
prim__shapeIndexPushFront : GCAnyPtr -> Int -> PrimIO ()

export
pushFront : HasIO io => ShapeIndex -> Nat -> io ()
pushFront (MkShapeIndex shapeIndex) value =
  primIO $ prim__shapeIndexPushFront shapeIndex (cast value)

%foreign (libxla "")  -- no C function!
prim__byteSizeOfElements : GCAnyPtr -> Bits64

export
byteSizeOfElements : Xla.Shape -> Bits64
byteSizeOfElements (MkShape shape) = prim__byteSizeOfElements shape

%foreign (libxla "MakeTupleShape")
prim__makeTupleShape : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

export
mkTupleShape : HasIO io => List Xla.Shape -> io Xla.Shape
mkTupleShape shapes = do
  MkShapeArray shapeArray <- mkShapeArray shapes
  shape <- primIO $ prim__makeTupleShape shapeArray (cast $ length shapes)
  shape <- onCollectAny shape Shape.delete
  pure (MkShape shape)

%foreign (libxla "MakeShape")
prim__mkShape : Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
mkShape : (HasIO io, Primitive dtype) => Types.Shape -> io Xla.Shape
mkShape shape = do
  let dtypeEnum = xlaIdentifier {dtype}
  MkIntArray shapeArrayPtr <- mkIntArray shape
  shapePtr <- primIO $ prim__mkShape dtypeEnum shapeArrayPtr (cast $ length shape)
  shapePtr <- onCollectAny shapePtr Shape.delete
  pure (MkShape shapePtr)
