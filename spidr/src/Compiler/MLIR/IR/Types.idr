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
module Compiler.MLIR.IR.Types

import Compiler.FFI
import Util

public export
data Type_ = MkType_ GCAnyPtr (GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ())

public export
data TypeArray = MkTypeArray GCAnyPtr

%foreign (libxla "sizeof_Type")
sizeofType : Bits64

%foreign (libxla "set_array_Type")
prim__setArrayType : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

export
mkTypeArray : HasIO io => List Type_ -> io TypeArray
mkTypeArray xs = do
  arr <- malloc (cast (length xs) * cast sizeofType)
  arr <- onCollectAny arr free
  traverse_ (\(idx, MkType_ x set) => primIO $ set arr (cast idx) x) (enumerate xs)
  pure (MkTypeArray arr)
