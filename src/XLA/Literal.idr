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
import XLA.XlaData
import XLA.FFI
import Types
import Util

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

public export
Literal : Type
Literal = Struct "Literal" []

export
interface XLAPrimitive dtype where
    primitiveType : PrimitiveType
    set : Literal -> Ptr Int -> dtype -> PrimIO ()
    get : Literal -> Ptr Int -> dtype

%foreign (libxla "Literal_new")
prim__allocLiteral : Ptr Int -> Int -> Int -> PrimIO Literal

%foreign (libxla "Literal_delete")
prim__Literal_delete : Literal -> PrimIO ()

export
delete : Literal -> IO ()
delete = primIO . prim__Literal_delete

populateLiteral : {rank : _} -> (shape : Shape {rank}) -> XLAPrimitive dtype =>
    Literal -> Array shape {dtype=dtype} -> IO ()
populateLiteral {rank} shape lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape {dtype=dtype} -> IO ()
    impl {a} [] acc_indices x = do
        idx_ptr <- mkIntArray acc_indices
        primIO $ set lit idx_ptr x
        free idx_ptr
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        traverse_ setArrays (enumerate xs) where
            setArrays : (Nat, Array rest {dtype=dtype}) -> IO ()
            setArrays (idx, xs') =
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                 in impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

export
mkLiteral : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}}
    -> Array shape {dtype=dtype} -> IO Literal
mkLiteral xs = do
    shape_ptr <- mkIntArray shape
    literal <- primIO $ prim__allocLiteral shape_ptr (cast rank) (cast $ primitiveType {dtype=dtype})
    populateLiteral shape literal xs
    free shape_ptr
    pure literal

%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : Literal -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_bool")
literalGetBool : Literal -> Ptr Int -> Int

export
XLAPrimitive Bool where
  primitiveType = PRED
  set lit idxs x = prim__literalSetBool lit idxs (if x then 1 else 0)
  get lit idxs = case literalGetBool lit idxs of
    0 => False
    1 => True
    x => (assert_total idris_crash) (
           "Internal error: expected 0 or 1 from XLA C API for boolean conversion, got " ++ show x
         )

%foreign (libxla "Literal_Set_double")
prim__literalSetDouble : Literal -> Ptr Int -> Double -> PrimIO ()

%foreign (libxla "Literal_Get_double")
literalGetDouble : Literal -> Ptr Int -> Double

export
XLAPrimitive Double where
  primitiveType = F64
  set = prim__literalSetDouble
  get = literalGetDouble

%foreign (libxla "Literal_Set_int")
prim__literalSetInt : Literal -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_int")
literalGetInt : Literal -> Ptr Int -> Int

export
XLAPrimitive Int where
  primitiveType = S32
  set = prim__literalSetInt
  get = literalGetInt

export
toArray : XLAPrimitive dtype => {shape : Shape} -> Literal -> Array shape {dtype=dtype}
toArray lit = impl {shapesSum=Refl} shape [] where
    impl : (remaining_shape : Vect r Nat)
        -> {a : _} -> {shapesSum : a + r = rank}
        -> (accumulated_indices : Vect a Nat)
        -> Array remaining_shape {dtype=dtype}
    impl [] acc = unsafePerformIO impl_io where
        impl_io : IO dtype
        impl_io = do
            idx_ptr <- mkIntArray acc
            let res = get lit idx_ptr
            free idx_ptr
            pure res
    impl (n :: rest) acc = map ((impl rest {shapesSum=Refl}) . (snoc acc)) (range n)
