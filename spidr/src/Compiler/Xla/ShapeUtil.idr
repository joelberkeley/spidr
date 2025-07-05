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
