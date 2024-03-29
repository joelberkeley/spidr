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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.Constants

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "MinValue")
prim__minValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "MinFiniteValue")
prim__minFiniteValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "MaxValue")
prim__maxValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
%foreign (libxla "MaxFiniteValue")
prim__maxFiniteValue : GCAnyPtr -> Int -> PrimIO AnyPtr
