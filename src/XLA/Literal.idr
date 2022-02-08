{--
Copyright 2021 Joel Berkeley

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
module XLA.Literal

import System.FFI

import Types
import Util
import XLA.XlaData
import XLA.FFI
import XLA.Shape
import XLA.ShapeUtil
import XLA.XlaData

export
interface Primitive dtype => LiteralRW dtype ty where
  rank_ : Nat
  shape_ : Shape {rank=rank_}
  set : GCAnyPtr -> List Nat -> ty -> IO ()
  get : GCAnyPtr -> List Nat -> ty

export
{len : Nat} -> LiteralRW dtype ty => LiteralRW dtype (Vect len ty) where
  rank_ = S (rank_ {dtype} {ty})
  shape_ = len :: shape_ {ty}
  set lit idxs xs = traverse_ (\(idx, x) => set {dtype} lit (idxs ++ [idx]) x) (enumerate xs)
  get lit idxs = map (\idx => get {dtype} lit (idxs ++ [idx])) (range len)

export
%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Literal_delete")
prim__Literal_delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__Literal_delete

export
mkLiteral : LiteralRW dtype ty => ty -> IO GCAnyPtr
mkLiteral xs = do
  xla_shape <- mkShape {dtype} (shape_ {dtype} {ty})
  literal <- primIO $ prim__allocLiteral xla_shape
  literal <- onCollectAny literal Literal.delete
  set {dtype} literal [] xs
  pure literal

%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> GCPtr Int -> Int

export
LiteralRW PRED Bool where
  -- dtype_ = PRED
  rank_ = 0
  shape_ = []
  set lit idxs x = primIO $ prim__literalSetBool lit !(mkIntArray idxs) (if x then 1 else 0)
  get lit idxs = case literalGetBool lit (unsafePerformIO $ mkIntArray idxs) of
    0 => False
    1 => True
    x => (assert_total idris_crash) (
           "Internal error: expected 0 or 1 from XLA C API for boolean conversion, got " ++ show x
         )

%foreign (libxla "Literal_Set_double")
prim__literalSetDouble : GCAnyPtr -> GCPtr Int -> Double -> PrimIO ()

%foreign (libxla "Literal_Get_double")
literalGetDouble : GCAnyPtr -> GCPtr Int -> Double

export
LiteralRW F64 Double where
  -- dtype_ = F64
  rank_ = 0
  shape_ = []
  set lit idxs x = primIO (prim__literalSetDouble lit !(mkIntArray idxs) x)
  get lit idxs = literalGetDouble lit (unsafePerformIO $ mkIntArray idxs)

%foreign (libxla "Literal_Set_int")
prim__literalSetInt : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_int")
literalGetInt : GCAnyPtr -> GCPtr Int -> Int

export
LiteralRW S32 Int where
  -- dtype_ = S32
  rank_ = 0
  shape_ = []
  set lit idxs x = primIO $ prim__literalSetInt lit (unsafePerformIO $ mkIntArray idxs) x
  get lit idxs = literalGetInt lit (unsafePerformIO $ mkIntArray idxs)

export
LiteralRW U32 Nat where
  -- dtype_ = U32
  rank_ = 0
  shape_ = []
  set lit idxs x = primIO $ prim__literalSetInt lit (unsafePerformIO $ mkIntArray idxs) (cast x)
  get lit idxs = cast (literalGetInt lit (unsafePerformIO $ mkIntArray idxs))

export
toArray : LiteralRW dtype ty => {shape : Shape} -> GCAnyPtr -> ty
toArray lit = get {dtype} lit (toList shape)
