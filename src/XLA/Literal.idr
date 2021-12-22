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
Literal = Struct "c__Literal" []

-- todo rename
export
interface XLAPrimitive dtype where
    primitiveType : PrimitiveType
    set : Literal -> Ptr Int -> dtype -> PrimIO ()
    get : Literal -> Ptr Int -> dtype

%foreign (libxla "c__Literal_new")
prim__allocLiteral : Ptr Int -> Int -> Int -> PrimIO Literal

%foreign (libxla "c__Literal_delete")
prim__delete : Literal -> PrimIO ()

export
delete : Literal -> IO ()
delete = primIO . prim__delete

populateLiteral : {rank : _} -> (shape : Shape {rank}) -> XLAPrimitive dtype =>
    Literal -> Array shape {dtype=dtype} -> IO ()
populateLiteral {rank} shape lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape {dtype=dtype} -> IO ()
    impl {a} [] acc_indices x = do
        idx_ptr <- mkShape acc_indices
        primIO $ set lit idx_ptr x
        freeShape idx_ptr
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        foldl setArrays (pure ()) (zip (range n) xs) where
            setArrays : IO () -> (Nat, Array rest {dtype=dtype}) -> IO ()
            setArrays prev_io (idx, xs') = do
                prev_io
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

export
mkLiteral : {rank : _} -> (shape : Shape {rank}) -> XLAPrimitive dtype =>
    Array shape {dtype=dtype} -> IO Literal
mkLiteral {rank} shape xs = do
    shape_ptr <- mkShape shape
    literal <- primIO $ prim__allocLiteral shape_ptr (cast rank) (cast $ primitiveType {dtype=dtype})
    populateLiteral shape literal xs
    freeShape shape_ptr
    pure literal

%foreign (libxla "c__Literal_Set_double")
prim__literalSetDouble : Literal -> Ptr Int -> Double -> PrimIO ()

%foreign (libxla "c__Literal_Get_double")
literalGetDouble : Literal -> Ptr Int -> Double

export
XLAPrimitive Double where
  primitiveType = F64
  set = prim__literalSetDouble
  get = literalGetDouble

%foreign (libxla "c__Literal_Set_int")
prim__literalSetInt : Literal -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "c__Literal_Get_int")
literalGetInt : Literal -> Ptr Int -> Int

export
XLAPrimitive Int where
  primitiveType = S32
  set = prim__literalSetInt
  get = literalGetInt

export
toArray : XLAPrimitive dtype => (shape : Shape) -> Literal -> Array shape {dtype=dtype}
toArray shape lit = impl {shapesSum=Refl} shape [] where
    impl : (remaining_shape : Vect r Nat)
        -> {a : _} -> {shapesSum : a + r = rank}
        -> (accumulated_indices : Vect a Nat)
        -> Array remaining_shape {dtype=dtype}
    impl [] acc = unsafePerformIO impl_io where
        impl_io : IO dtype
        impl_io = do
            idx_ptr <- mkShape acc
            let res = get lit idx_ptr
            freeShape idx_ptr
            pure res
    impl (n :: rest) acc = map ((impl rest {shapesSum=Refl}) . (snoc acc)) (range n)
