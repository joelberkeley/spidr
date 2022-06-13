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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.XlaBuilder

import System.FFI

import Compiler.Xla.Prim.Util

namespace XlaBuilder
  export
  %foreign (libxla "XlaBuilder_delete")
  prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "XlaBuilder_new")
prim__mkXlaBuilder : String -> PrimIO AnyPtr

export
%foreign (libxla "CreateSubBuilder")
prim__createSubBuilder : GCAnyPtr -> String -> PrimIO AnyPtr

export
%foreign (libxla "XlaBuilder_Build")
prim__build : GCAnyPtr -> AnyPtr

export
%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : GCAnyPtr -> GCAnyPtr -> String

namespace XlaOp
  export
  %foreign (libxla "XlaOp_delete")
  prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "sizeof_XlaOp")
sizeOfXlaOp : Int

export
%foreign (libxla "set_array_XlaOp")
prim__setArrayXlaOp : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

export
%foreign (libxla "Parameter")
prim__parameter : GCAnyPtr -> Int -> GCAnyPtr -> String -> PrimIO AnyPtr

export
%foreign (libxla "ConstantLiteral")
prim__constantLiteral : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "BroadcastInDim")
prim__broadcastInDim : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Reshape")
prim__reshape : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Slice")
prim__slice : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "ConcatInDim")
prim__concatInDim : GCAnyPtr -> GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Select")
prim__select : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Tuple")
prim__tuple : GCAnyPtr -> GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "GetTupleElement")
prim__getTupleElement : GCAnyPtr -> Int -> PrimIO AnyPtr

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
%foreign (libxla "Dot")
prim__dot : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "TriangularSolve")
prim__triangularSolve : GCAnyPtr -> GCAnyPtr -> Int -> Int -> Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Cholesky")
prim__cholesky : GCAnyPtr -> Int -> PrimIO AnyPtr

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
%foreign (libxla "Exp")
prim__exp : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Floor")
prim__floor : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Ceil")
prim__ceil : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Log")
prim__log : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Logistic")
prim__logistic : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Cos")
prim__cos : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Sin")
prim__sin : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Tanh")
prim__tanh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Sqrt")
prim__sqrt : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Pow")
prim__pow : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "ConvertElementType")
prim__convertElementType : GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "BitcastConvertType")
prim__bitcastConvertType : GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Neg")
prim__neg : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Transpose")
prim__transpose : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Sort")
prim__sort : GCAnyPtr -> Int -> GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

export
%foreign (libxla "Map")
prim__map :
  GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr -> GCPtr Int -> Int -> AnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "RngBitGenerator")
prim__rngBitGenerator : Int -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Conditional")
prim__conditional : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr
