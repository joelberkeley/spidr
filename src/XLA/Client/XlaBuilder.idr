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

import XLA.FFI
import XLA.Literal
import XLA.XlaData
import Types
import System.FFI
import Util

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

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
 - XlaOp
 -
 -}

export
XlaOp : Type
XlaOp = Struct "c__XlaOp" []

-- todo one solution to this is to make everything AnyPtr, which we'll do for the finalisers anyway
%foreign (libxla "c__ConstantLiteral")
constantLiteral : XlaBuilder -> Literal -> XlaOp

export
const : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}} -> 
    XlaBuilder -> Array shape {dtype=dtype} -> IO XlaOp
const {dtype} {shape} builder arr =
    do literal <- mkLiteral shape arr
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

export
eval : XLAPrimitive dtype => {shape : _} -> XlaOp -> IO (Array shape {dtype=dtype})
eval op = map (toArray shape) (primIO $ prim__eval op)

%foreign (libxla "c__XlaBuilder_OpToString")
export
opToString : XlaBuilder -> XlaOp -> String

%foreign (libxla "c__XlaOp_operator_add")
export
(+) : XlaOp -> XlaOp -> XlaOp
