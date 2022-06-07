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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Shape

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "sizeof_Shape")
sizeofShape : Int

export
%foreign (libxla "set_array_Shape")
prim__setArrayShape : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

export
%foreign (libxla "Shape_delete")
prim__delete : AnyPtr -> PrimIO ()