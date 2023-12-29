{--
Copyright 2023 Joel Berkeley

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
module Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Service.GPU.Runtime.Support

import System.FFI

import Compiler.Xla.Prim.Util

export
%foreign (libxla "DotDimensionNumbers_new")
prim__dotDimensionNumbersNew : PrimIO AnyPtr

export
%foreign (libxla "DotDimensionNumbers_delete")
prim__dotDimensionNumbersDelete : AnyPtr -> PrimIO ()

export
%foreign (libxla "DotDimensionNumbers_add_lhs_contracting_dimensions")
prim__addLhsContractingDimensions : GCAnyPtr -> Int -> PrimIO ()

export
%foreign (libxla "DotDimensionNumbers_add_rhs_contracting_dimensions")
prim__addRhsContractingDimensions : GCAnyPtr -> Int -> PrimIO ()

export
%foreign (libxla "DotDimensionNumbers_add_lhs_batch_dimensions")
prim__addLhsBatchDimensions : GCAnyPtr -> Int -> PrimIO ()

export
%foreign (libxla "DotDimensionNumbers_add_rhs_batch_dimensions")
prim__addRhsBatchDimensions : GCAnyPtr -> Int -> PrimIO ()
