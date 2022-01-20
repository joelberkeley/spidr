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

%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Literal_delete")
prim__Literal_delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "Literal_shape")
Literal_shape : GCAnyPtr -> AnyPtr

export
delete : AnyPtr -> IO ()
delete = primIO . prim__Literal_delete

populateLiteral : {rank : _} -> (shape : Shape {rank}) -> XLAPrimitive dtype =>
    GCAnyPtr -> Array shape dtype -> IO ()
populateLiteral {rank} shape lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape dtype -> IO ()
    impl {a} [] acc_indices x = do
        idx_ptr <- mkIntArray acc_indices
        primIO $ set lit idx_ptr x
        free idx_ptr
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        traverse_ setArrays (enumerate xs) where
            setArrays : (Nat, Array rest dtype) -> IO ()
            setArrays (idx, xs') =
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                 in impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

export
mkLiteral : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}}
    -> Array shape dtype -> GCAnyPtr
mkLiteral xs = unsafePerformIO $ do
    xla_shape <- mkShape {dtype=dtype} shape
    literal <- primIO $ prim__allocLiteral xla_shape
    literal <- onCollectAny literal Literal.delete
    populateLiteral shape literal xs
    pure literal

export
%foreign (libxla "Literal_ToString")
toString : GCAnyPtr -> String

%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> Ptr Int -> Int

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
prim__literalSetDouble : GCAnyPtr -> Ptr Int -> Double -> PrimIO ()

%foreign (libxla "Literal_Get_double")
literalGetDouble : GCAnyPtr -> Ptr Int -> Double

export
XLAPrimitive Double where
  primitiveType = F64
  set = prim__literalSetDouble
  get = literalGetDouble

%foreign (libxla "Literal_Set_int")
prim__literalSetInt : GCAnyPtr -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_int")
literalGetInt : GCAnyPtr -> Ptr Int -> Int

export
XLAPrimitive Int where
  primitiveType = S32
  set = prim__literalSetInt
  get = literalGetInt

export
toArray : XLAPrimitive dtype => {shape : Types.Shape} -> GCAnyPtr -> Array shape dtype
toArray lit = impl {shapesSum=Refl} shape [] where
    impl : (remaining_shape : Vect r Nat)
        -> {a : _} -> {shapesSum : a + r = rank}
        -> (accumulated_indices : Vect a Nat)
        -> Array remaining_shape dtype
    impl [] acc = unsafePerformIO impl_io where
        impl_io : IO dtype
        impl_io = do
            idx_ptr <- mkIntArray acc
            let res = get lit idx_ptr
            free idx_ptr
            pure res
    impl (n :: rest) acc = map ((impl rest {shapesSum=Refl}) . (snoc acc)) (range n)
