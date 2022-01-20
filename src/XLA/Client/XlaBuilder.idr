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
module XLA.Client.XlaBuilder

import Data.Vect
import System.FFI

import XLA.Shape
import XLA.Client.XlaComputation
import XLA.FFI
import XLA.Literal
import XLA.XlaData
import Types
import Util

{-
 -
 - XlaBuilder
 -
 -}

namespace XlaBuilder
    %foreign (libxla "XlaBuilder_delete")
    prim__delete : AnyPtr -> PrimIO ()

    export
    delete : AnyPtr -> IO ()
    delete = primIO . prim__delete

%foreign (libxla "XlaBuilder_new")
prim__mkXlaBuilder : String -> PrimIO AnyPtr

export
mkXlaBuilder : String -> IO GCAnyPtr
mkXlaBuilder computation_name = do
    builder <- primIO (prim__mkXlaBuilder computation_name)
    onCollectAny builder XlaBuilder.delete

export
%foreign (libxla "XlaBuilder_Build")
build : GCAnyPtr -> XlaComputation

export
%foreign (libxla "XlaBuilder_GetShapePtr")
getShapePtr : AnyPtr -> GCAnyPtr -> AnyPtr

export
%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : GCAnyPtr -> GCAnyPtr -> String

{-
 -
 - XlaOp
 -
 -}

export
%foreign (libxla "test")
test : PrimIO ()

export
%foreign (libxla "sizeof_XlaOp")
sizeof_xlaOp : Int

export
%foreign (libxla "set_array_XlaOp")
prim__setArrayXlaOp : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

%foreign (libxla "XlaOp_delete")
prim__XlaOp_delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "XlaOp_Builder")
builder : GCAnyPtr -> AnyPtr

export
collectXlaOp : AnyPtr -> IO GCAnyPtr
collectXlaOp op = onCollectAny op $ primIO . prim__XlaOp_delete

export
%foreign (libxla "XlaOp_print")
print : GCAnyPtr -> PrimIO ()

export
%foreign (libxla "Parameter")
parameter : GCAnyPtr -> Int -> GCAnyPtr -> String -> AnyPtr

export
%foreign (libxla "ConstantLiteral")
constantLiteral : GCAnyPtr -> GCAnyPtr -> AnyPtr

export
%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> Ptr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Eq")
prim__eq : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Ne")
prim__ne : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Ge")
prim__ge : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Gt")
prim__gt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Lt")
prim__lt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Le")
prim__le : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Call")
prim__call : GCAnyPtr -> XlaComputation -> AnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Add")
prim__add : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Sub")
prim__sub : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Mul")
prim__mul : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Div")
prim__div : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "And")
prim__and : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Or")
prim__or : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Not")
prim__not : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Abs")
prim__abs : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Neg")
prim__neg : GCAnyPtr -> PrimIO AnyPtr

{-
 -
 - XlaOp and GCAnyPtr wrapper
 -
 -}

public export
data RawTensor = MkRawTensor (GCAnyPtr -> IO GCAnyPtr)

export
broadcast : {n : _} -> RawTensor -> Vect n Nat -> RawTensor
broadcast (MkRawTensor f) broadcast_sizes = MkRawTensor $ \builder =>
    do broadcast_sizes_ptr <- mkIntArray broadcast_sizes
       op <- primIO $ prim__broadcast !(f builder) broadcast_sizes_ptr (cast n)
       op <- collectXlaOp op
       free broadcast_sizes_ptr
       pure op

%foreign (libxla "BroadcastInDim")
prim__broadcastInDim : GCAnyPtr -> Ptr Int -> Int -> Ptr Int -> Int -> PrimIO AnyPtr

export
broadcastInDim : {r : _} -> RawTensor -> Shape {rank=r} -> Shape {rank=r} -> RawTensor
broadcastInDim (MkRawTensor f) ods bcd = MkRawTensor $ \builder =>
    do ods_ptr <- mkIntArray ods
       bcd_ptr <- mkIntArray bcd
       op <- primIO $ prim__broadcastInDim !(f builder) ods_ptr (cast r) bcd_ptr (cast r)
       op <- collectXlaOp op
       free ods_ptr
       free bcd_ptr
       pure op
