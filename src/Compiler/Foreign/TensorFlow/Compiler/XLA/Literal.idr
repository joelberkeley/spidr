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
module Compiler.Foreign.TensorFlow.Compiler.XLA.Literal

import System.FFI

import Compiler.Foreign.Util

export
%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Literal_delete")
prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

export
%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> GCPtr Int -> Int

export
%foreign (libxla "Literal_Set_double")
prim__literalSetDouble : GCAnyPtr -> GCPtr Int -> Double -> PrimIO ()

export
%foreign (libxla "Literal_Get_double")
literalGetDouble : GCAnyPtr -> GCPtr Int -> Double

export
%foreign (libxla "Literal_Set_int")
prim__literalSetInt : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

export
%foreign (libxla "Literal_Get_int")
literalGetInt : GCAnyPtr -> GCPtr Int -> Int
