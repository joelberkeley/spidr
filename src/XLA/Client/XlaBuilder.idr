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
prim__mkXlaBuilderImpl : String -> PrimIO AnyPtr

export
prim__mkXlaBuilder : String -> IO GCAnyPtr
prim__mkXlaBuilder computation_name = do
  builder <- primIO (prim__mkXlaBuilderImpl computation_name)
  onCollectAny builder XlaBuilder.delete

%foreign (libxla "XlaBuilder_Build")
prim__buildImpl : GCAnyPtr -> AnyPtr

export
prim__build : GCAnyPtr -> IO GCAnyPtr
prim__build builder = onCollectAny (prim__buildImpl builder) XlaComputation.delete

%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : GCAnyPtr -> GCAnyPtr -> String

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
parameter : GCAnyPtr -> Int -> GCAnyPtr -> String -> AnyPtr

%foreign (libxla "ConstantLiteral")
constantLiteral : GCAnyPtr -> GCAnyPtr -> AnyPtr

%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

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
data RawTensor = MkRawTensor (GCAnyPtr -> IO GCAnyPtr)

export
const : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}}
        -> Array shape dtype -> RawTensor
const arr = MkRawTensor $ \builder => do
  literal <- mkLiteral arr
  collectXlaOp (constantLiteral builder literal)

export
toString : RawTensor -> IO String
toString (MkRawTensor f) = do
  builder <- prim__mkXlaBuilder ""
  pure (prim__opToString builder !(f builder))

export
broadcast : {n : _} -> RawTensor -> Vect n Nat -> RawTensor
broadcast (MkRawTensor f) broadcast_sizes = MkRawTensor $ \builder =>
    do broadcast_sizes_ptr <- mkIntArray broadcast_sizes
       primIO (prim__broadcast !(f builder) broadcast_sizes_ptr (cast n)) >>= collectXlaOp

%foreign (libxla "BroadcastInDim")
prim__broadcastInDim : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
broadcastInDim : {r : _} -> RawTensor -> Shape {rank=r} -> Shape {rank=r} -> RawTensor
broadcastInDim (MkRawTensor f) ods bcd = MkRawTensor $ \builder =>
    do ods_ptr <- mkIntArray ods
       bcd_ptr <- mkIntArray bcd
       primIO (prim__broadcastInDim !(f builder) ods_ptr (cast r) bcd_ptr (cast r)) >>= collectXlaOp

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
