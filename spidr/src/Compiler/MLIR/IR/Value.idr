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
module Compiler.MLIR.IR.Value

import Util
import Compiler.FFI

public export
data Value = MkValue GCAnyPtr (GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ())

public export
data ValueArray = MkValueArray GCAnyPtr

%foreign (libxla "sizeof_Value")
sizeofValue : Bits64

export
mkValueArray :  HasIO io => List Value -> io ValueArray
mkValueArray xs = do
  arr <- malloc (cast (length xs) * cast sizeofValue)
  arr <- onCollectAny arr free
  traverse_ (\(idx, MkValue x set) => primIO $ set arr (cast idx) x) (enumerate xs)
  pure (MkValueArray arr)

public export
data BlockArgument = MkBlockArgument GCAnyPtr

export
%foreign (libxla "BlockArgument_delete")
prim__deleteBlockArgument : AnyPtr -> PrimIO ()

%foreign (libxla "set_array_BlockArgument")
prim__setArrayBlockArgument : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

export
Cast BlockArgument Value where
  cast (MkBlockArgument ptr) = MkValue ptr prim__setArrayBlockArgument

public export
data OpResult = MkOpResult GCAnyPtr

export
%foreign (libxla "OpResult_delete")
prim__deleteOpResult : AnyPtr -> PrimIO ()

%foreign (libxla "set_array_OpResult")
prim__setArrayOpResult : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

export
Cast OpResult Value where
  cast (MkOpResult ptr) = MkValue ptr prim__setArrayOpResult
