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
 - XlaOp
 -
 -}

%foreign (libxla "c__XlaOp_delete")
prim__delete_XlaOp : AnyPtr -> PrimIO ()

%foreign (libxla "c__ConstantLiteral")
constantLiteral : AnyPtr -> Literal -> AnyPtr

%foreign (libxla "c__XlaOp_operator_add")
prim__XlaOp_operator_add : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

{-
 -
 - XlaBuilder
 -
 -}

%foreign (libxla "c__XlaBuilder_delete")
prim__delete_XlaBuilder : AnyPtr -> PrimIO ()

%foreign (libxla "c__XlaBuilder_new")
prim__mkXlaBuilder : String -> AnyPtr

%foreign (libxla "c__XlaBuilder_OpToString")
prim__opToString : AnyPtr -> GCAnyPtr -> String

{-
 -
 - XlaOp and XlaBuilder wrapper
 -
 -}

export data RawTensor = MkRawTensor (AnyPtr -> IO GCAnyPtr)

export
const : XLAPrimitive dtype => {shape : _} -> Array shape {dtype} -> RawTensor
const arr = MkRawTensor $ \builder =>
    do literal <- mkLiteral arr
       let op = onCollectAny (constantLiteral builder literal) $ primIO . prim__delete_XlaOp
       delete literal
       op

export
opToString : RawTensor -> IO String
opToString (MkRawTensor f) =
  do let builder = prim__mkXlaBuilder ""
         str = pure $ prim__opToString builder !(f builder)
     primIO $ prim__delete_XlaBuilder builder
     str

%foreign (libxla "eval")
prim__eval : GCAnyPtr -> PrimIO Literal

export
eval : XLAPrimitive dtype => {shape : _} -> RawTensor -> IO (Array shape {dtype})
eval (MkRawTensor f) =
    do let builder = prim__mkXlaBuilder ""
       lit <- primIO $ prim__eval !(f builder)
       let arr = toArray lit
       delete lit
       primIO $ prim__delete_XlaBuilder builder
       pure arr

export
(+) : RawTensor -> RawTensor -> RawTensor
(MkRawTensor l) + (MkRawTensor r) = MkRawTensor $ \builder =>
    do new_op <- primIO $ prim__XlaOp_operator_add !(l builder) !(r builder)
       onCollectAny new_op $ primIO . prim__delete_XlaOp
