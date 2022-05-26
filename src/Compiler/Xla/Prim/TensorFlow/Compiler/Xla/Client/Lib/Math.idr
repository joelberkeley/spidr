{--
Copyright 2022 Joel Berkeley

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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.Math

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "Square")
prim__square : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Reciprocal")
prim__reciprocal : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Acos")
prim__acos : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Asin")
prim__asin : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Atan")
prim__atan : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Tan")
prim__tan : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Acosh")
prim__acosh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Asinh")
prim__asinh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Atanh")
prim__atanh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Cosh")
prim__cosh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Sinh")
prim__sinh : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Erf")
prim__erf : GCAnyPtr -> PrimIO AnyPtr
