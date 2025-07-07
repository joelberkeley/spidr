/*
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
*/
#include "xla/hlo/builder/lib/math.h"

#include "../xla_builder.h"

extern "C" {
    XlaOp* Square(XlaOp& x) { return unaryOp(xla::Square, x); }
    XlaOp* Reciprocal(XlaOp& x) { return unaryOp(xla::Reciprocal, x); }
    XlaOp* Acos(XlaOp& x) { return unaryOp(xla::Acos, x); }
    XlaOp* Asin(XlaOp& x) { return unaryOp(xla::Asin, x); }
    XlaOp* Atan(XlaOp& x) { return unaryOp(xla::Atan, x); }
    XlaOp* Tan(XlaOp& x) { return unaryOp(xla::Tan, x); }
    XlaOp* Acosh(XlaOp& x) { return unaryOp(xla::Acosh, x); }
    XlaOp* Asinh(XlaOp& x) { return unaryOp(xla::Asinh, x); }
    XlaOp* Atanh(XlaOp& x) { return unaryOp(xla::Atanh, x); }
    XlaOp* Cosh(XlaOp& x) { return unaryOp(xla::Cosh, x); }
    XlaOp* Sinh(XlaOp& x) { return unaryOp(xla::Sinh, x); }
    XlaOp* Erf(XlaOp& x) { return unaryOp(xla::Erf, x); }
}
