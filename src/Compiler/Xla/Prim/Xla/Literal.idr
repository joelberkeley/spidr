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
module Compiler.Xla.Prim.Xla.Literal

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "Literal_untyped_data")
prim__literalUntypedData : GCAnyPtr -> AnyPtr

export
%foreign (libxla "Literal_size_bytes")
prim__literalSizeBytes : GCAnyPtr -> Int

export
%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Literal_delete")
prim__delete : AnyPtr -> PrimIO ()

export
%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int -> PrimIO ()

export
%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int

export
%foreign (libxla "Literal_Set_double")
prim__literalSetDouble : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Double -> PrimIO ()

export
%foreign (libxla "Literal_Get_double")
literalGetDouble : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Double

export
%foreign (libxla "Literal_Set_int32_t")
prim__literalSetInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int -> PrimIO ()

export
%foreign (libxla "Literal_Get_int32_t")
literalGetInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Int

export
%foreign (libxla "Literal_Set_uint32_t")
prim__literalSetUInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits32 -> PrimIO ()

export
%foreign (libxla "Literal_Get_uint32_t")
literalGetUInt32t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits32

export
%foreign (libxla "Literal_Set_uint64_t")
prim__literalSetUInt64t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits64 -> PrimIO ()

export
%foreign (libxla "Literal_Get_uint64_t")
literalGetUInt64t : GCAnyPtr -> GCPtr Int -> Int -> GCAnyPtr -> Bits64
