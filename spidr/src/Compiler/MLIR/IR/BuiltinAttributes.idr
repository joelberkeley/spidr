{--
Copyright 2025 Joel Berkeley

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
module Compiler.MLIR.IR.BuiltinAttributes

import Compiler.MLIR.IR.BuiltinTypeInterfaces
import Compiler.FFI

public export
data DenseElementsAttr = MkDenseElementsAttr GCAnyPtr

%foreign (libxla "DenseElementsAttr_delete")
prim__deleteDenseElementsAttr : AnyPtr -> PrimIO ()

%foreign (libxla "DenseElementsAttr_get")
prim__denseElementsAttrGet : GCAnyPtr -> Double -> PrimIO AnyPtr

namespace DenseElementsAttr
  export
  get : HasIO io => ShapedType -> Double -> io DenseElementsAttr
  get (MkShapedType st) value = do
    attr <- primIO $ prim__denseElementsAttrGet st value
    attr <- onCollectAny attr (primIO . prim__deleteDenseElementsAttr)
    pure (MkDenseElementsAttr attr)
