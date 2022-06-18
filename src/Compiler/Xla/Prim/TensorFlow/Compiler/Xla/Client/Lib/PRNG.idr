
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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.PRNG

import System.FFI

import Compiler.Xla.Prim.Util

public export
RngOutput : Type
RngOutput = Struct "RngOutput" [("value", AnyPtr), ("state", AnyPtr)]

export
%foreign (libxla "delete_RngOutput")
prim__delete : RngOutput -> PrimIO ()

export
%foreign (libxla "UniformFloatingPointDistribution")
prim__uniformFloatingPointDistribution:
  GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO RngOutput
