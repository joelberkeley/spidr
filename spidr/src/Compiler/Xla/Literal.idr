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
||| For internal spidr use only.
module Compiler.Xla.Literal

import Compiler.Xla.Shape
import Compiler.Xla.ShapeUtil
import Compiler.Xla.XlaData
import Compiler.FFI
import Types

namespace Xla
  public export
  data Literal : Type where
    MkLiteral : GCAnyPtr -> Literal

%foreign (libxla "Literal_delete")
prim__delete : AnyPtr -> PrimIO ()

%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

export
allocLiteral : HasIO io => Xla.Shape -> io Literal
allocLiteral (MkShape shape) = do
  litPtr <- primIO $ prim__allocLiteral shape
  litPtr <- onCollectAny litPtr (primIO . prim__delete)
  pure (MkLiteral litPtr)

export
%foreign (libxla "Literal_size_bytes")
prim__literalSizeBytes : GCAnyPtr -> Int

export
%foreign (libxla "Literal_untyped_data")
prim__literalUntypedData : GCAnyPtr -> AnyPtr

namespace Bool
  %foreign (libxla "Literal_Set_bool")
  prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int -> PrimIO ()

  export
  set : Literal -> List Nat -> ShapeIndex -> Bool -> IO ()
  set (MkLiteral lit) idxs (MkShapeIndex shapeIndex) value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $
      prim__literalSetBool lit idxsArrayPtr (cast $ length idxs) shapeIndex (boolToCInt value)

  %foreign (libxla "Literal_Get_bool")
  literalGetBool : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int

  export
  get : Literal -> List Nat -> ShapeIndex -> Bool
  get (MkLiteral lit) idxs (MkShapeIndex shapeIndex) = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cIntToBool $ literalGetBool lit idxsArrayPtr (cast $ length idxs) shapeIndex

namespace Double
  %foreign (libxla "Literal_Set_double")
  prim__literalSetDouble : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Double -> PrimIO ()

  export
  set : Literal -> List Nat -> ShapeIndex -> Double -> IO ()
  set (MkLiteral lit) idxs (MkShapeIndex shapeIndex) value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetDouble lit idxsArrayPtr (cast $ length idxs) shapeIndex value

  %foreign (libxla "Literal_Get_double")
  literalGetDouble : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Double

  export
  get : Literal -> List Nat -> ShapeIndex -> Double
  get (MkLiteral lit) idxs (MkShapeIndex shapeIndex) = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ literalGetDouble lit idxsArrayPtr (cast $ length idxs) shapeIndex

namespace Int32t
  %foreign (libxla "Literal_Set_int32_t")
  prim__literalSetInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int -> PrimIO ()

  export
  set : Literal -> List Nat -> ShapeIndex -> Int32 -> IO ()
  set (MkLiteral lit) idxs (MkShapeIndex shapeIndex) value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetInt32t lit idxsArrayPtr (cast $ length idxs) shapeIndex (cast value)

  %foreign (libxla "Literal_Get_int32_t")
  literalGetInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int

  export
  get : Literal -> List Nat -> ShapeIndex -> Int32
  get (MkLiteral lit) idxs (MkShapeIndex shapeIndex) = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetInt32t lit idxsArrayPtr (cast $ length idxs) shapeIndex

namespace UInt32t
  %foreign (libxla "Literal_Set_uint32_t")
  prim__literalSetUInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits32 -> PrimIO ()

  export
  set : Literal -> List Nat -> ShapeIndex -> Nat -> IO ()
  set (MkLiteral lit) idxs (MkShapeIndex shapeIndex) value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetUInt32t lit idxsArrayPtr (cast $ length idxs) shapeIndex (cast value)

  %foreign (libxla "Literal_Get_uint32_t")
  literalGetUInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits32

  export
  get : Literal -> List Nat -> ShapeIndex -> Nat
  get (MkLiteral lit) idxs (MkShapeIndex shapeIndex) = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetUInt32t lit idxsArrayPtr (cast $ length idxs) shapeIndex

namespace UInt64t
  %foreign (libxla "Literal_Set_uint64_t")
  prim__literalSetUInt64t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits64 -> PrimIO ()

  export
  set : Literal -> List Nat -> ShapeIndex -> Nat -> IO ()
  set (MkLiteral lit) idxs (MkShapeIndex shapeIndex) value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetUInt64t lit idxsArrayPtr (cast $ length idxs) shapeIndex (cast value)

  %foreign (libxla "Literal_Get_uint64_t")
  literalGetUInt64t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits64

  export
  get : Literal -> List Nat -> ShapeIndex -> Nat
  get (MkLiteral lit) idxs (MkShapeIndex shapeIndex) = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetUInt64t lit idxsArrayPtr (cast $ length idxs) shapeIndex
