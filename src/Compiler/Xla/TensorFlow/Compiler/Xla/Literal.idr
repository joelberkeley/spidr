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
module Compiler.Xla.TensorFlow.Compiler.Xla.Literal

import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.Util
import Types

namespace Xla
  public export
  data Literal : Type where
    MkLiteral : GCAnyPtr -> Literal

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
  export
  set : Literal -> List Nat -> Bool -> IO ()
  set (MkLiteral lit) idxs value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetBool lit idxsArrayPtr (if value then 1 else 0)

  export
  get : Literal -> List Nat -> Bool
  get (MkLiteral lit) idxs = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cIntToBool $ literalGetBool lit idxsArrayPtr

namespace Double
  export
  set : Literal -> List Nat -> Double -> IO ()
  set (MkLiteral lit) idxs value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetDouble lit idxsArrayPtr value

  export
  get : Literal -> List Nat -> Double
  get (MkLiteral lit) idxs = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ literalGetDouble lit idxsArrayPtr

namespace Int32t
  export
  set : Literal -> List Nat -> Int32 -> IO ()
  set (MkLiteral lit) idxs value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetInt32t lit idxsArrayPtr (cast value)

  export
  get : Literal -> List Nat -> Int32
  get (MkLiteral lit) idxs = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetInt32t lit idxsArrayPtr

namespace UInt32t
  export
  set : Literal -> List Nat -> Nat -> IO ()
  set (MkLiteral lit) idxs value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetUInt32t lit idxsArrayPtr (cast value)

  export
  get : Literal -> List Nat -> Nat
  get (MkLiteral lit) idxs = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetUInt32t lit idxsArrayPtr

namespace UInt64t
  export
  set : Literal -> List Nat -> Nat -> IO ()
  set (MkLiteral lit) idxs value = do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    primIO $ prim__literalSetUInt64t lit idxsArrayPtr (cast value)

  export
  get : Literal -> List Nat -> Nat
  get (MkLiteral lit) idxs = unsafePerformIO $ do
    MkIntArray idxsArrayPtr <- mkIntArray idxs
    pure $ cast $ literalGetUInt64t lit idxsArrayPtr
