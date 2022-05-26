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
module Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Math

import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.Math
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder

export
square : HasIO io => XlaOp -> io XlaOp
square = unaryOp prim__square

export
reciprocal : HasIO io => XlaOp -> io XlaOp
reciprocal = unaryOp prim__reciprocal

export
acos : HasIO io => XlaOp -> io XlaOp
acos = unaryOp prim__acos

export
asin : HasIO io => XlaOp -> io XlaOp
asin = unaryOp prim__asin

export
atan : HasIO io => XlaOp -> io XlaOp
atan = unaryOp prim__atan

export
tan : HasIO io => XlaOp -> io XlaOp
tan = unaryOp prim__tan

export
acosh : HasIO io => XlaOp -> io XlaOp
acosh = unaryOp prim__acosh

export
asinh : HasIO io => XlaOp -> io XlaOp
asinh = unaryOp prim__asinh

export
atanh : HasIO io => XlaOp -> io XlaOp
atanh = unaryOp prim__atanh

export
cosh : HasIO io => XlaOp -> io XlaOp
cosh = unaryOp prim__cosh

export
sinh : HasIO io => XlaOp -> io XlaOp
sinh = unaryOp prim__sinh

export
erf : HasIO io => XlaOp -> io XlaOp
erf = unaryOp prim__erf
