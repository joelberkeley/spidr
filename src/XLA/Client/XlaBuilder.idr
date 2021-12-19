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
prim__allocIntArray : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "index_i32")
prim__indexI32 : AnyPtr -> Int -> Int

%foreign (libxla "index_f64")
prim__indexF64 : AnyPtr -> Int -> Double

%foreign (libxla "index_void_ptr")
prim__indexArray : AnyPtr -> Int -> AnyPtr

rangeTo : (n : Nat) -> Vect n Nat
rangeTo Z = []
rangeTo (S n) = snoc (rangeTo n) (S n)

setElems : AnyPtr -> (shape : Shape {rank=S _}) -> (dtype : Type)
    -> Array shape {dtype=dtype} -> IO ()
setElems array_ptr [n] dtype xs = foldr f (pure ()) (zip (rangeTo n) xs) where
    f : (Nat, dtype) -> IO () -> IO ()
    f (sidx, x) a = do a
                       primIO $ setter array_ptr (cast $ pred sidx) x

        where
        setter : AnyPtr -> Int -> dtype -> PrimIO ()
        setter = case dtype of
            Int => prim__array_set_I32
            Double => prim__array_set_F64
            _ => ?rhs'
setElems array_ptr (n :: r :: est) dtype xs = foldr f (pure ()) (zip (rangeTo n) xs)

    where
    f : (Nat, Array (r :: est) {dtype=dtype}) -> IO () -> IO ()
    f (sidx, x) a = do a
                       setElems (prim__indexArray array_ptr (cast $ pred sidx)) (r :: est) dtype x

export
putArray : {r : _} -> (shape : Shape {rank=S r}) -> (dtype : Type)
    -> Array shape {dtype=dtype} -> IO AnyPtr
putArray {r} shape dtype xs = do
    shape_ptr <- primIO (prim__allocShape $ cast (S r))
    foldr (setDims shape_ptr) (pure ()) (zip (rangeTo (S r)) shape)
    array_ptr <- primIO $ prim__allocIntArray shape_ptr
    setElems array_ptr shape dtype xs
    pure array_ptr
        where setDims : AnyPtr -> (Nat, Nat) -> IO () -> IO ()
              setDims ptr (sidx, dim) acc =
                 do acc
                    primIO (prim__array_set_I32 ptr (cast $ pred sidx) (cast dim))

%foreign (libxla "test_put")
export
test_put : AnyPtr -> PrimIO ()

export
const : XlaBuilder -> Array shape -> XlaOp

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

%foreign (libxla "eval_i32")
prim__eval_i32 : XlaOp -> PrimIO Int

%foreign (libxla "eval_f64")
prim__eval_f64 : XlaOp -> PrimIO Double

%foreign (libxla "eval_array")
prim__eval_array : XlaOp -> PrimIO AnyPtr

export
getArray : (shape : Shape {rank=S _}) -> (dtype : Type) -> AnyPtr -> Array shape {dtype=dtype}
getArray [n] dtype ptr = map (indexByType ptr . cast . pred) (rangeTo n) where
    indexByType : AnyPtr -> Int -> dtype
    indexByType = case dtype of
        -- todo use interfaces rather than pattern matching on types, then can possibly erase
        -- dtype
        Int => prim__indexI32
        Double => prim__indexF64
        _ => ?rhs
getArray (n :: r :: est) dtype ptr =
    map ((getArray (r :: est) dtype) . (prim__indexArray ptr . cast . pred)) (rangeTo n)

export
eval : {dtype : _} -> {shape : Shape} -> XlaOp -> IO (Array shape {dtype=dtype})
eval {shape=[]} op = eval_scalar op where
    eval_scalar : XlaOp -> IO dtype
    eval_scalar = case dtype of
        Int => primIO . prim__eval_i32
        Double => primIO . prim__eval_f64
eval {shape=n :: rest} op = map (getArray (n :: rest) dtype) (primIO $ prim__eval_array op)
