/*
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
