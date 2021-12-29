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
import Data.Vect

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

{-
 -
 - XlaBuilder
 -
 -}

XlaBuilder : Type
XlaBuilder = Struct "XlaBuilder" []

%foreign (libxla "XlaBuilder_delete")
prim__XlaBuilder_delete : XlaBuilder -> PrimIO ()

delete : XlaBuilder -> IO ()
delete = primIO . prim__XlaBuilder_delete

%foreign (libxla "XlaBuilder_new")
prim__mkXlaBuilder : String -> PrimIO XlaBuilder

%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : XlaBuilder -> GCAnyPtr -> String

{-
 -
 - XlaOp
 -
 -}

%foreign (libxla "XlaOp_delete")
prim__XlaOp_delete : AnyPtr -> PrimIO ()

collectXlaOp : AnyPtr -> IO GCAnyPtr
collectXlaOp op = onCollectAny op $ primIO . prim__XlaOp_delete

%foreign (libxla "ConstantLiteral")
constantLiteral : XlaBuilder -> Literal -> AnyPtr

%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> Ptr Int -> Int -> PrimIO AnyPtr

%foreign (libxla "Eq")
prim__eq : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Neq")
prim__neq : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Add")
prim__add : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

{-
 -
 - XlaOp and XlaBuilder wrapper
 -
 -}

export data RawTensor = MkRawTensor (XlaBuilder -> IO GCAnyPtr)

export
const : XLAPrimitive dtype => {rank : _} -> {shape : Shape {rank}}
    -> Array shape {dtype} -> RawTensor
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

%foreign (libxla "eval")
prim__eval : GCAnyPtr -> PrimIO Literal

export
eval : XLAPrimitive dtype => {shape : _} -> RawTensor -> IO (Array shape {dtype})
eval (MkRawTensor f) =
    do builder <- primIO (prim__mkXlaBuilder "")
       lit <- primIO $ prim__eval !(f builder)
       let arr = toArray lit
       delete lit
       delete builder
       pure arr

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

mkBinOp : (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr) -> RawTensor -> RawTensor -> RawTensor
mkBinOp f (MkRawTensor l) (MkRawTensor r) = MkRawTensor $ \builder =>
    do op <- primIO $ f !(l builder) !(r builder)
       collectXlaOp op

export
eq : RawTensor -> RawTensor -> RawTensor
eq = mkBinOp prim__eq

export
neq : RawTensor -> RawTensor -> RawTensor
neq = mkBinOp prim__neq

export
add : RawTensor -> RawTensor -> RawTensor
add = mkBinOp prim__add
