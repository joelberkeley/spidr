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

%foreign (libxla "CreateSubBuilder")
prim__createSubBuilderImpl : GCAnyPtr -> String -> PrimIO AnyPtr

export
prim__createSubBuilder : GCAnyPtr -> String -> IO GCAnyPtr
prim__createSubBuilder builder computation_name = do
  sub_builder <- primIO (prim__createSubBuilderImpl builder computation_name)
  onCollectAny sub_builder XlaBuilder.delete

%foreign (libxla "XlaBuilder_Build")
prim__buildImpl : GCAnyPtr -> AnyPtr

export
prim__build : GCAnyPtr -> IO GCAnyPtr
prim__build builder = onCollectAny (prim__buildImpl builder) XlaComputation.delete

export
%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : GCAnyPtr -> GCAnyPtr -> String

{-
 -
 - XlaOp
 -
 -}

namespace XlaOp
  %foreign (libxla "XlaOp_delete")
  prim__delete : AnyPtr -> PrimIO ()

  export
  delete : AnyPtr -> IO ()
  delete = primIO . prim__delete

%foreign (libxla "sizeof_XlaOp")
sizeOfXlaOp : Int

%foreign (libxla "set_array_XlaOp")
prim__setArrayXlaOp : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

export
mkXlaOpArray : List GCAnyPtr -> IO GCAnyPtr
mkXlaOpArray ops = do
  arr <- malloc (cast (length ops) * sizeOfXlaOp)
  traverse_ (\(idx, op) =>
    primIO $ prim__setArrayXlaOp arr (cast idx) op) (enumerate (fromList ops))
  onCollectAny arr free

export
%foreign (libxla "Parameter")
parameter : GCAnyPtr -> Int -> GCAnyPtr -> String -> AnyPtr

export
%foreign (libxla "ConstantLiteral")
constantLiteral : GCAnyPtr -> GCAnyPtr -> AnyPtr

export
%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "BroadcastInDim")
prim__broadcastInDim : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

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
%foreign (libxla "Max")
prim__max : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Min")
prim__min : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

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
%foreign (libxla "Reduce")
prim__reduce : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Abs")
prim__abs : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Neg")
prim__neg : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Map")
prim__map : GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr
            -> GCPtr Int -> Int -> AnyPtr -> Int -> PrimIO AnyPtr
