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
module Compiler.TensorFlow.Compiler.Xla.Literal

import System.FFI

import Compiler.FFI
import Compiler.TensorFlow.Compiler.Xla.Shape
import Compiler.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.TensorFlow.Compiler.Xla.XlaData
import Types

%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

namespace Xla
  public export
  data Literal : Type where
    MkLiteral : GCAnyPtr -> Literal

%foreign (libxla "Literal_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__delete

export
allocLiteral : HasIO io => Primitive dtype => Types.Shape -> io Literal
allocLiteral shape = do
  MkShape shapePtr <- mkShape {dtype} shape
  litPtr <- primIO $ prim__allocLiteral shapePtr
  litPtr <- onCollectAny litPtr Literal.delete
  pure (MkLiteral litPtr)

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
