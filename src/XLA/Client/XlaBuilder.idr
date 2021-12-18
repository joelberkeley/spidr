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
export
const : XlaBuilder -> Int -> XlaOp

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

%foreign (libxla "index_i32")
prim__indexI32 : AnyPtr -> Int -> Int

%foreign (libxla "index_f64")
prim__indexF64 : AnyPtr -> Int -> Double

%foreign (libxla "index_void_ptr")
prim__indexArray : AnyPtr -> Int -> AnyPtr

export
%foreign (libxla "arr")
nums : AnyPtr

rangeTo : (n : Nat) -> Vect n Nat
rangeTo Z = []
rangeTo (S n) = snoc (rangeTo n) (S n)

export
build_array : (shape : Shape {rank=S _}) -> (dtype : Type) -> AnyPtr -> Array shape {dtype=dtype}
build_array [n] dtype ptr = map (indexByType ptr . cast . pred) (rangeTo n) where
    indexByType : AnyPtr -> Int -> dtype
    indexByType = case dtype of
        -- todo use interfaces rather than pattern matching on types, then can possibly erase
        -- dtype
        Int => prim__indexI32
        Double => prim__indexF64
        _ => ?rhs
build_array (n :: r :: est) dtype ptr =
    map ((build_array (r :: est) dtype) . (prim__indexArray ptr . cast . pred)) (rangeTo n)

export
eval : {dtype : _} -> {shape : Shape} -> XlaOp -> IO (Array shape {dtype=dtype})
eval {shape=[]} op = eval_scalar op where
    eval_scalar : XlaOp -> IO dtype
    eval_scalar = case dtype of
        Int => primIO . prim__eval_i32
        Double => primIO . prim__eval_f64
eval {shape=n :: rest} op = map (build_array (n :: rest) dtype) (primIO $ prim__eval_array op)
