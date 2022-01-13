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

XlaBuilder : Type
XlaBuilder = Struct "XlaBuilder" []

%foreign (libxla "XlaBuilder_delete")
prim__XlaBuilder_delete : XlaBuilder -> PrimIO ()

export
delete : XlaBuilder -> IO ()
delete = primIO . prim__XlaBuilder_delete

export
%foreign (libxla "XlaBuilder_new")
prim__mkXlaBuilder : String -> PrimIO XlaBuilder

export
%foreign (libxla "XlaBuilder_Build")
build : XlaBuilder -> XlaComputation

%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : XlaBuilder -> GCAnyPtr -> String

{-
 -
 - XlaOp
 -
 -}

%foreign (libxla "XlaOp_delete")
prim__XlaOp_delete : AnyPtr -> PrimIO ()

export
collectXlaOp : AnyPtr -> IO GCAnyPtr
collectXlaOp op = onCollectAny op $ primIO . prim__XlaOp_delete

export
%foreign (libxla "Parameter")
parameter : XlaBuilder -> Int -> Shape.Shape -> String -> AnyPtr

%foreign (libxla "ConstantLiteral")
constantLiteral : XlaBuilder -> Literal -> AnyPtr

%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> Ptr Int -> Int -> PrimIO AnyPtr

%foreign (libxla "Eq")
prim__eq : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Ne")
prim__ne : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Ge")
prim__ge : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Gt")
prim__gt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Lt")
prim__lt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Le")
prim__le : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Add")
prim__add : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Sub")
prim__sub : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Mul")
prim__mul : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Div")
prim__div : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "And")
prim__and : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Or")
prim__or : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Not")
prim__not : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Abs")
prim__abs : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Neg")
prim__neg : GCAnyPtr -> PrimIO AnyPtr

{-
 -
 - XlaOp and XlaBuilder wrapper
 -
 -}

public export
data RawTensor = MkRawTensor (XlaBuilder -> IO GCAnyPtr)

export
const : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}}
    -> Array shape dtype -> RawTensor
const arr = MkRawTensor $ \builder =>
    do literal <- mkLiteral arr
       op <- collectXlaOp (constantLiteral builder literal)
       delete literal
       pure op

export
toString : RawTensor -> IO String
toString (MkRawTensor f) =
  do builder <- primIO (prim__mkXlaBuilder "")
     let str = prim__opToString builder !(f builder)
     delete builder
     pure str

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

unaryOp : (GCAnyPtr -> PrimIO AnyPtr) -> RawTensor -> RawTensor
unaryOp f (MkRawTensor operand) = MkRawTensor $ \builder =>
    do op <- primIO $ f !(operand builder)
       collectXlaOp op

binOp : (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr) -> RawTensor -> RawTensor -> RawTensor
binOp f (MkRawTensor l) (MkRawTensor r) = MkRawTensor $ \builder =>
    do op <- primIO $ f !(l builder) !(r builder)
       collectXlaOp op

export
eq : RawTensor -> RawTensor -> RawTensor
eq = binOp prim__eq

export
ne : RawTensor -> RawTensor -> RawTensor
ne = binOp prim__ne

export
ge : RawTensor -> RawTensor -> RawTensor
ge = binOp prim__ge

export
gt : RawTensor -> RawTensor -> RawTensor
gt = binOp prim__gt

export
lt : RawTensor -> RawTensor -> RawTensor
lt = binOp prim__lt

export
le : RawTensor -> RawTensor -> RawTensor
le = binOp prim__le

export
add : RawTensor -> RawTensor -> RawTensor
add = binOp prim__add

export
sub : RawTensor -> RawTensor -> RawTensor
sub = binOp prim__sub

export
mul : RawTensor -> RawTensor -> RawTensor
mul = binOp prim__mul

export
div : RawTensor -> RawTensor -> RawTensor
div = binOp prim__div

export
and : RawTensor -> RawTensor -> RawTensor
and = binOp prim__and

export
or : RawTensor -> RawTensor -> RawTensor
or = binOp prim__or

export
not : RawTensor -> RawTensor
not = unaryOp prim__not

export
abs : RawTensor -> RawTensor
abs = unaryOp prim__abs

export
neg : RawTensor -> RawTensor
neg = unaryOp prim__neg
