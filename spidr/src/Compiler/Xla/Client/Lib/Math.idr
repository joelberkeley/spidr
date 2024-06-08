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
||| For internal spidr use only.
module Compiler.Xla.Client.Lib.Math

import Compiler.FFI
import Compiler.Xla.Client.XlaBuilder

%foreign (libxla "Square")
prim__square : GCAnyPtr -> PrimIO AnyPtr

export
square : HasIO io => XlaOp -> io XlaOp
square = unaryOp prim__square

%foreign (libxla "Reciprocal")
prim__reciprocal : GCAnyPtr -> PrimIO AnyPtr

export
reciprocal : HasIO io => XlaOp -> io XlaOp
reciprocal = unaryOp prim__reciprocal

%foreign (libxla "Acos")
prim__acos : GCAnyPtr -> PrimIO AnyPtr

export
acos : HasIO io => XlaOp -> io XlaOp
acos = unaryOp prim__acos

%foreign (libxla "Asin")
prim__asin : GCAnyPtr -> PrimIO AnyPtr

export
asin : HasIO io => XlaOp -> io XlaOp
asin = unaryOp prim__asin

%foreign (libxla "Atan")
prim__atan : GCAnyPtr -> PrimIO AnyPtr

export
atan : HasIO io => XlaOp -> io XlaOp
atan = unaryOp prim__atan

%foreign (libxla "Tan")
prim__tan : GCAnyPtr -> PrimIO AnyPtr

export
tan : HasIO io => XlaOp -> io XlaOp
tan = unaryOp prim__tan

%foreign (libxla "Acosh")
prim__acosh : GCAnyPtr -> PrimIO AnyPtr

export
acosh : HasIO io => XlaOp -> io XlaOp
acosh = unaryOp prim__acosh

%foreign (libxla "Asinh")
prim__asinh : GCAnyPtr -> PrimIO AnyPtr

export
asinh : HasIO io => XlaOp -> io XlaOp
asinh = unaryOp prim__asinh

%foreign (libxla "Atanh")
prim__atanh : GCAnyPtr -> PrimIO AnyPtr

export
atanh : HasIO io => XlaOp -> io XlaOp
atanh = unaryOp prim__atanh

%foreign (libxla "Cosh")
prim__cosh : GCAnyPtr -> PrimIO AnyPtr

export
cosh : HasIO io => XlaOp -> io XlaOp
cosh = unaryOp prim__cosh

%foreign (libxla "Sinh")
prim__sinh : GCAnyPtr -> PrimIO AnyPtr

export
sinh : HasIO io => XlaOp -> io XlaOp
sinh = unaryOp prim__sinh

%foreign (libxla "Erf")
prim__erf : GCAnyPtr -> PrimIO AnyPtr

export
erf : HasIO io => XlaOp -> io XlaOp
erf = unaryOp prim__erf
