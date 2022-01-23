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

export
interface Primitive dtype => LiteralPrimitiveRW dtype ty | ty where
    set : GCAnyPtr -> GCPtr Int -> ty -> PrimIO ()
    get : GCAnyPtr -> GCPtr Int -> ty

export
%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Literal_delete")
prim__Literal_delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__Literal_delete

populateLiteral : {rank : _} -> (shape : Shape {rank}) -> LiteralPrimitiveRW dtype ty =>
    GCAnyPtr -> Array shape ty -> IO ()
populateLiteral {rank} shape lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape ty -> IO ()
    impl {a} [] acc_indices x = primIO $ set {dtype} lit !(mkIntArray acc_indices) x
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        traverse_ setArrays (enumerate xs) where
            setArrays : (Nat, Array rest ty) -> IO ()
            setArrays (idx, xs') =
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                 in impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

export
mkLiteral : LiteralPrimitiveRW dtype ty => {rank : _} -> {shape : Shape {rank}}
            -> Array shape ty -> IO GCAnyPtr
mkLiteral xs = do
    xla_shape <- mkShape {dtype} shape
    literal <- primIO $ prim__allocLiteral xla_shape
    literal <- onCollectAny literal Literal.delete
    populateLiteral {dtype} shape literal xs
    pure literal

%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> GCPtr Int -> Int

export
LiteralPrimitiveRW PRED Bool where
  set lit idxs x = prim__literalSetBool lit idxs (if x then 1 else 0)
  get lit idxs = case literalGetBool lit idxs of
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
LiteralPrimitiveRW F64 Double where
  set = prim__literalSetDouble
  get = literalGetDouble

%foreign (libxla "Literal_Set_int")
prim__literalSetInt : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_int")
literalGetInt : GCAnyPtr -> GCPtr Int -> Int

export
LiteralPrimitiveRW S32 Int where
  set = prim__literalSetInt
  get = literalGetInt

export
LiteralPrimitiveRW U32 Nat where
  set lit idx x = prim__literalSetInt lit idx (cast x)
  get = cast .: literalGetInt

export
toArray : LiteralPrimitiveRW dtype ty => {shape : Shape} -> GCAnyPtr -> Array shape ty
toArray lit = impl {shapesSum=Refl} shape [] where
    impl : (remaining_shape : Vect r Nat)
        -> {a : _} -> {shapesSum : a + r = rank}
        -> (accumulated_indices : Vect a Nat)
        -> Array remaining_shape ty
    impl [] acc = unsafePerformIO $ map (get {dtype} lit) (mkIntArray acc)
    impl (n :: rest) acc = map ((impl rest {shapesSum=Refl}) . (snoc acc)) (range n)
