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

%foreign (libxla "to_int")
prim__to_int : Literal -> Int

%foreign (libxla "to_double")
prim__to_double : Literal -> Double

%foreign (libxla "to_array_int")
prim__to_array_int : Literal -> AnyPtr

%foreign (libxla "array_to_literal")
prim__array_int_to_literal : AnyPtr -> AnyPtr -> Int -> Literal

indicesForLength : (n : Nat) -> Vect n Nat
indicesForLength Z = []
indicesForLength (S n) = snoc (indicesForLength n) n

setElems : AnyPtr -> (shape : Shape {rank=S _}) -> (dtype : Type)
    -> Array shape {dtype=dtype} -> IO ()
setElems array_ptr [n] dtype xs = foldl f (pure ()) (zip (indicesForLength n) xs) where
    f : IO () -> (Nat, dtype) -> IO ()
    f a (idx, x) = do a
                      putStrLn $ "setElems.f " ++ show idx
                      res <- primIO $ setter array_ptr (cast idx) x
                      putStrLn $ "setElems.f' " ++ show idx
        where
        setter : AnyPtr -> Int -> dtype -> PrimIO ()
        setter = case dtype of
            Int => prim__array_set_I32
            Double => prim__array_set_F64
            _ => ?rhs'
setElems array_ptr (n :: r :: est) dtype xs = foldl ff (pure ()) (zip (indicesForLength n) xs)
    where
    ff : IO () -> (Nat, Array (r :: est) {dtype=dtype}) -> IO ()
    ff a (idx, x) = do a
                       putStrLn $ "setElems.ff idx " ++ (show idx)
                       setElems (prim__indexArray array_ptr (cast idx)) (r :: est) dtype x

putArray : {r : _} -> (shape : Shape {rank=S r}) -> (dtype : Type)
    -> Array shape {dtype=dtype} -> IO (AnyPtr, AnyPtr)
putArray {r} shape dtype xs = do
    putStrLn "putArray ... allocate shape array"
    shape_ptr <- primIO (prim__allocShape $ cast (S r))
    putStrLn "putArray ... populate shape array"
    -- foldl (setDims shape_ptr) (pure ()) (zip (indicesForLength (S r)) shape)
    setElems shape_ptr [S r] Int (map cast shape)
    putStrLn "putArray ... allocate actual array"
    array_ptr <- primIO $ prim__allocIntArray shape_ptr (cast (S r))
    putStrLn "putArray ... populate actual array"
    setElems array_ptr shape dtype xs
    putStrLn "putArray ... return array pointer"
    pure (shape_ptr, array_ptr)
        -- where setDims : AnyPtr -> IO () -> (Nat, Nat) -> IO ()
        --       setDims ptr acc (idx, dim) =
        --          do acc
        --             primIO (prim__array_set_I32 ptr (cast idx) (cast dim))

%foreign (libxla "constant")
prim__const : XlaBuilder -> Literal -> XlaOp

export
const : {r : _} -> {shape : Shape {rank=S r}} -> {dtype : _} -> XlaBuilder -> Array shape {dtype=dtype} -> IO XlaOp
const {r} {dtype} {shape} builder arr =
    do putStrLn "const ... construct C arrays"
       (shape_ptr, arr_ptr) <- putArray shape dtype arr
       putStrLn "const ... build XlaOp"
       res <- pure $ prim__const builder $ prim__array_int_to_literal arr_ptr shape_ptr (cast (S r)) 
       putStrLn "const ... return XlaOp"
       pure res

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
eval : {dtype : _} -> {shape : Shape} -> XlaOp -> IO (Array shape {dtype=dtype})
eval {shape} op = map (to_idris_type shape) (primIO $ prim__eval op) where
    to_idris_type : (shape : Shape) -> Literal -> Array shape {dtype=dtype}
    to_idris_type [] = case dtype of
            Int => prim__to_int
            Double => prim__to_double
    to_idris_type (n :: rest) = (getArray (n :: rest) dtype) . prim__to_array_int
