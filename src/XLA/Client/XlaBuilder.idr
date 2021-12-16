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

%foreign (libxla "eval")
prim__eval : XlaOp -> PrimIO AnyPtr

indexArray : AnyPtr -> AnyPtr

indexU32 : AnyPtr -> Nat

indexU64 : AnyPtr -> Nat

indexI32 : AnyPtr -> Int

indexI64 : AnyPtr -> Integer

indexF32 : AnyPtr -> Double  -- hmmm

indexF64 : AnyPtr -> Double

rangeTo : (n : Nat) -> Vect n Nat
rangeTo Z = []
rangeTo (S n) = snoc (rangeTo n) (S n)

points : Vect m (n ** Vect n Nat) => Array [] {dtype=Nat}

build_array : (shape : Shape) -> AnyPtr -> Array shape {dtype=dtype}
build_array {dtype} shape x =
    let axes = the (Vect _ (n ** Vect n Nat)) $ map (\n => (_ ** rangeTo n)) shape
     in ?rhs

export
eval_int : {shape : Shape} -> XlaOp -> IO (Array [] {dtype=Int})
eval_int op = map (build_array [] {dtype=Int}) (primIO $ prim__eval op)
