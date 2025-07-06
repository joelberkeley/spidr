{--
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
||| For internal spidr use only.
module Compiler.Xla.HLO.Builder.Lib.Math

import Compiler.FFI
import Compiler.Xla.HLO.Builder.XlaBuilder

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
