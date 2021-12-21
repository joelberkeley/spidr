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

import XLA.XlaData
import Types
import System.FFI

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

xla_crash : Show a => a -> b
xla_crash x = (assert_total idris_crash) $ "Fatal: XLA C API produced unexpected value " ++ show x

{-
 -
 - XlaBuilder
 -
 -}

export
XlaBuilder : Type
XlaBuilder = Struct "c__XlaBuilder" []

%foreign (libxla "c__XlaBuilder_new")
export
mkXlaBuilder : String -> XlaBuilder

namespace XlaBuilder
    %foreign (libxla "c__XlaBuilder_delete")
    prim__delete : XlaBuilder -> PrimIO ()

    export
    delete : XlaBuilder -> IO ()
    delete = primIO . prim__delete

%foreign (libxla "c__XlaBuilder_name")
export
name : XlaBuilder -> String

{-
 -
 - Array shape
 -
 -}

%foreign (libxla "alloc_shape")
prim__allocShape : Int -> PrimIO (Ptr Int)

%foreign (libxla "free_shape")
prim__free_shape : Ptr Int -> PrimIO ()

%foreign (libxla "set_shape_idx")
prim__set_shape_idx : Ptr Int -> Int -> Int -> PrimIO ()

indicesForLength : (n : Nat) -> Vect n Nat
indicesForLength Z = []
indicesForLength (S n) = snoc (indicesForLength n) n

mkShape : {rank : _} -> Shape {rank} -> IO (Ptr Int)
mkShape {rank} xs = do
    ptr <- primIO $ prim__allocShape (cast rank)
    foldl (writeElem ptr) (pure ()) (zip (indicesForLength rank) xs)
    pure ptr where
        writeElem : Ptr Int -> IO () -> (Nat, Nat) -> IO ()
        writeElem ptr prev_io (idx, x) = do
            prev_io
            primIO $ prim__set_shape_idx ptr (cast idx) (cast x)

{-
 -
 - Literal
 -
 -}

export
Literal : Type
Literal = Struct "c__Literal" []

%foreign (libxla "c__Literal_new")
prim__Literal_new : Ptr Int -> Int -> Int -> PrimIO Literal

namespace Literal
    %foreign (libxla "c__Literal_delete")
    prim__delete : Literal -> PrimIO ()

    export
    delete : Literal -> IO ()
    delete = primIO . prim__delete

%foreign (libxla "c__Literal_Set_int")
prim__Literal_Set_int : Literal -> Ptr Int -> Int -> PrimIO ()

%foreign (libxla "c__Literal_Set_double")
prim__Literal_Set_double : Literal -> Ptr Int -> Double -> PrimIO ()

populateLiteral : {rank : _} -> (shape : Shape {rank}) -> (dtype : Type)
    -> Literal -> Array shape {dtype=dtype} -> IO ()
populateLiteral {rank} shape dtype lit arr = impl {shapesSum=Refl} shape [] arr where
    impl : {a : _} -> (rem_shape : Shape {rank=r}) -> (acc_indices : Shape {rank=a})
        -> {shapesSum : a + r = rank} -> Array rem_shape {dtype=dtype} -> IO ()
    impl {a} [] acc_indices x = do
        idx_ptr <- mkShape acc_indices
        let setter = case dtype of
                Int => prim__Literal_Set_int
                Double => prim__Literal_Set_double
        primIO $ setter lit idx_ptr x
        primIO $ prim__free_shape idx_ptr
    impl {shapesSum} {r=S r'} {a} (n :: rest) acc_indices xs =
        foldl setArrays (pure ()) (zip (indicesForLength n) xs) where
            setArrays : IO () -> (Nat, Array rest {dtype=dtype}) -> IO ()
            setArrays prev_io (idx, xs') = do
                prev_io
                let shapesSum' = rewrite plusSuccRightSucc a r' in shapesSum
                impl {shapesSum=shapesSum'} rest (snoc acc_indices idx) xs'

mkLiteral : {rank : _} -> (shape : Shape {rank}) -> (dtype : Type) -> ScalarLike dtype =>
    Array shape {dtype=dtype} -> IO Literal
mkLiteral {rank} shape dtype xs = do
    shape_ptr <- mkShape shape
    literal <- primIO $ prim__Literal_new shape_ptr (cast rank) (cast $ primitiveType {dtype=dtype})
    populateLiteral shape dtype literal xs
    primIO $ prim__free_shape shape_ptr
    pure literal

%foreign (libxla "c__Literal_Get_int")
Literal_Get_int : Literal -> Ptr Int -> Int

%foreign (libxla "c__Literal_Get_double")
Literal_Get_double : Literal -> Ptr Int -> Double

%foreign (libxla "to_int")
to_int : Literal -> Int

%foreign (libxla "to_double")
to_double : Literal -> Double

{-
 -
 - XlaOp
 -
 -}

export
XlaOp : Type
XlaOp = Struct "c__XlaOp" []

%foreign (libxla "c__ConstantLiteral")
constantLiteral : XlaBuilder -> Literal -> XlaOp

export
const : {rank : _} -> {shape : Shape {rank}} -> {dtype : _} -> ScalarLike dtype =>
    XlaBuilder -> Array shape {dtype=dtype} -> IO XlaOp
const {dtype} {shape} builder arr =
    do literal <- mkLiteral shape dtype arr
       let op = constantLiteral builder literal
       delete literal
       pure op

namespace XlaOp
    %foreign (libxla "c__XlaOp_delete")
    prim__delete : XlaOp -> PrimIO ()

    export
    delete : XlaOp -> IO ()
    delete = primIO . prim__delete

%foreign (libxla "eval")
prim__eval : XlaOp -> PrimIO Literal

toArray : (shape : Shape) -> (dtype : Type) -> Literal -> Array shape {dtype=dtype}
toArray shape dtype lit = impl {shapesSum=Refl} shape [] where
    impl : (remaining_shape : Vect r Nat)
        -> {a : _} -> {shapesSum : a + r = rank}
        -> (accumulated_indices : Vect a Nat)
        -> Array remaining_shape {dtype=dtype}
    impl [] acc = unsafePerformIO impl_io where  -- todo unsafePerformIO
        impl_io : IO dtype
        impl_io = do
            idx_ptr <- mkShape acc
            let setter = case dtype of
                    Int => Literal_Get_int
                    Double => Literal_Get_double
                res = setter lit idx_ptr
            primIO $ prim__free_shape idx_ptr
            pure res
    impl (n :: rest) acc = map ((impl rest {shapesSum=Refl}) . (snoc acc)) (indicesForLength n)

export
eval : {dtype : _} -> {shape : _} -> XlaOp -> IO (Array shape {dtype=dtype})
eval op = map (toArray shape dtype) (primIO $ prim__eval op)

%foreign (libxla "c__XlaBuilder_OpToString")
export
opToString : XlaBuilder -> XlaOp -> String

%foreign (libxla "c__XlaOp_operator_add")
export
(+) : XlaOp -> XlaOp -> XlaOp
