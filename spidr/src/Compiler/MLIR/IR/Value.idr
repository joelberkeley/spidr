{--
Copyright 2024 Joel Berkeley

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
data Value = MkValue GCAnyPtr

public export
data ValueArray = MkValueArray GCAnyPtr

-- this needs to set the ptr, not the Value itself
%foreign (libxla "set_array_Value")
prim__setArrayValue : GCAnyPtr -> Bits64 -> GCAnyPtr -> PrimIO ()

export
mkValueArray : HasIO io => List Value -> io ValueArray
mkValueArray xs = do
  -- i can only guess this is an array of Value* (not Value), else how else do we create this?
  ptr <- malloc (cast (length xs) * cast sizeofPtr)
  ptr <- onCollectAny ptr free
  traverse_ (\(idx, MkValue x) => primIO $ prim__setArrayValue ptr (cast idx) (cast x)) (enumerate xs)
  pure (MkValueArray ptr)

public export
data BlockArgument = MkBlockArgument GCAnyPtr

export
Cast BlockArgument Value where
  cast (MkBlockArgument a) = MkValue a

public export
data OpResult = MkOpResult GCAnyPtr

export
Cast OpResult Value where
  cast (MkOpResult r) = MkValue r
