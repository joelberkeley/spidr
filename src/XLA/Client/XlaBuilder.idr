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
||| This module contains the Idris API to XLA.
module XLA.Client.XlaBuilder

import XLA
import Types
import System.FFI

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

xla_crash : Show a => a -> b
xla_crash x = (assert_total idris_crash) $ "Fatal: XLA C API produced unexpected value " ++ show x

export
XlaBuilder : Type
XlaBuilder = Struct "c__XlaBuilder" []

%foreign (libxla "c__XlaBuilder_new")
export
mkXlaBuilder : String -> XlaBuilder

%foreign (libxla "c__XlaBuilder_name")
export
name : XlaBuilder -> String

namespace XlaBuilder
    %foreign (libxla "c__XlaBuilder_del")
    prim__delete : XlaBuilder -> PrimIO ()

    export
    delete : XlaBuilder -> IO ()
    delete = primIO . prim__delete

export
XlaOp : Type
XlaOp = Struct "c__XlaOp" []

%foreign (libxla "c__ConstantR0")
prim__constant_r0 : XlaBuilder -> Int -> XlaOp

%foreign (libxla "alloc_shape")
prim__allocShape : Int -> PrimIO AnyPtr

%foreign (libxla "array_set_i32")
prim__array_set_I32 : AnyPtr -> Int -> Int -> PrimIO ()

%foreign (libxla "array_set_f64")
prim__array_set_F64 : AnyPtr -> Int -> Double -> PrimIO ()

%foreign (libxla "array_alloc_i32")
prim__allocIntArray : AnyPtr -> Int -> PrimIO AnyPtr

%foreign (libxla "index_i32")
prim__indexI32 : AnyPtr -> Int -> Int

%foreign (libxla "index_f64")
prim__indexF64 : AnyPtr -> Int -> Double

%foreign (libxla "index_void_ptr")
prim__indexArray : AnyPtr -> Int -> AnyPtr

export
Literal : Type
Literal = Struct "c__Literal" []

%foreign (libxla "c__Literal_Set_int")
prim__Literal_Set_int : Literal -> AnyPtr -> Int -> PrimIO ()

%foreign (libxla "c__Literal_Get_int")
Literal_Get_int : Literal -> AnyPtr -> Int

%foreign (libxla "to_int")
to_int : Literal -> Int

%foreign (libxla "to_double")
to_double : Literal -> Double

%foreign (libxla "to_array_int")
prim__to_array_int : Literal -> AnyPtr

%foreign (libxla "array_to_literal")
prim__array_int_to_literal : AnyPtr -> AnyPtr -> Int -> Literal

%foreign (libxla "c__Literal_new")
prim__Literal_new : AnyPtr -> Int -> PrimIO Literal

indicesForLength : (n : Nat) -> Vect n Nat
indicesForLength Z = []
indicesForLength (S n) = snoc (indicesForLength n) n

mkShape : {rank : _} -> Shape {rank} -> IO AnyPtr
mkShape {rank} xs = do
    ptr <- primIO $ prim__allocShape (cast rank)
    foldl (f ptr) (pure ()) (zip (indicesForLength rank) xs)
    pure ptr where
        f : AnyPtr -> IO () -> (Nat, Nat) -> IO ()
        f ptr prev_io (idx, x) = do prev_io; primIO $ prim__array_set_I32 ptr (cast idx) (cast x)

-- todo merge this with prim__Literal_new into a single mkLiteral function?
populateLiteral : {rank : _} -> (shape : Shape {rank}) -> (dtype : Type)
    -> Literal -> Array shape {dtype=dtype} -> IO ()
populateLiteral {rank} shape dtype lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape {dtype=dtype} -> IO ()
    impl {a} [] acc_indices x = do
        idx_ptr <- mkShape acc_indices
        primIO $ case dtype of
            Int => prim__Literal_Set_int lit idx_ptr x
            _ => ?other_dtypes
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        foldl setArrays (pure ()) (zip (indicesForLength n) xs) where
            setArrays : IO () -> (Nat, Array rest {dtype=dtype}) -> IO ()
            setArrays prev_io (idx, xs') = do
                prev_io
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

mkLiteral : {rank : _} -> (shape : Shape {rank}) -> (dtype : Type)
    -> Array shape {dtype=dtype} -> IO Literal
mkLiteral {rank} shape dtype xs = do
    shape_ptr <- mkShape shape
    literal_ptr <- primIO $ prim__Literal_new shape_ptr (cast rank)
    populateLiteral shape dtype literal_ptr xs
    pure literal_ptr

%foreign (libxla "c__ConstantLiteral")
constantLiteral : XlaBuilder -> Literal -> XlaOp

export
const : {rank : _} -> {shape : Shape {rank}} -> {dtype : _}
    -> XlaBuilder -> Array shape {dtype=dtype} -> IO XlaOp
const {dtype} {shape} builder arr =
    do literal_ptr <- mkLiteral shape dtype arr
       pure $ constantLiteral builder literal_ptr

namespace XlaOp
    %foreign (libxla "c__XlaOp_del")
    prim__delete : XlaOp -> PrimIO ()

    export
    delete : XlaOp -> IO ()
    delete = primIO . prim__delete

%foreign (libxla "c__XlaBuilder_OpToString")
export
opToString : XlaBuilder -> XlaOp -> String

%foreign (libxla "c__XlaOp_operator_add")
export
(+) : XlaOp -> XlaOp -> XlaOp

%foreign (libxla "eval")
prim__eval : XlaOp -> PrimIO Literal

-- todo rewrite in terms of literal
getArray : (shape : Shape {rank=S _}) -> (dtype : Type) -> AnyPtr -> Array shape {dtype=dtype}
getArray [n] dtype ptr = map (indexByType ptr . cast) (indicesForLength n) where
    indexByType : AnyPtr -> Int -> dtype
    indexByType = case dtype of
        -- todo use interfaces rather than pattern matching on types, then can possibly erase
        -- dtype
        Int => prim__indexI32
        Double => prim__indexF64
        _ => ?rhs
getArray (n :: r :: est) dtype ptr =
    map ((getArray (r :: est) dtype) . (prim__indexArray ptr . cast)) (indicesForLength n)

export
eval : {dtype : _} -> {shape : _} -> XlaOp -> IO (Array shape {dtype=dtype})
eval op = map (to_idris_type shape) (primIO $ prim__eval op) where
    to_idris_type : (shape : Shape) -> Literal -> Array shape {dtype=dtype}
    to_idris_type [] = case dtype of
            Int => to_int
            Double => to_double
    to_idris_type (n :: rest) = (getArray (n :: rest) dtype) . prim__to_array_int
