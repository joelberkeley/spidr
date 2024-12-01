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
||| For internal spidr use only.
module Compiler.FFI

import public System.FFI
import Util

public export
libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libc_xla"

public export
data CharArray = MkCharArray (Ptr Char) Bits64

namespace CharArray
  export
  free : HasIO io => CharArray -> io ()
  free (MkCharArray arr _) = free $ prim__forgetPtr arr

export
%foreign (libxla "string_delete")
prim__stringDelete : AnyPtr -> PrimIO ()

export
%foreign (libxla "string_data")
prim__stringData : AnyPtr -> PrimIO $ Ptr Char

export
%foreign (libxla "string_size")
prim__stringSize : AnyPtr -> Bits64

export
%foreign (libxla "idx")
prim__index : Int -> AnyPtr -> AnyPtr

export
stringToCharArray : HasIO io => AnyPtr -> io MkCharArray
stringToCharArray str = do
  data' <- primIO $ prim__stringData str
  let size = prim__stringSize str
  primIO $ prim__stringDelete str
  pure (MkCharArray data' size)

export
cIntToBool : Int -> Bool
cIntToBool 0 = False
cIntToBool 1 = True
cIntToBool x =
  let msg = "Internal error: expected 0 or 1 from XLA C API for boolean conversion, got " ++ show x
  in (assert_total idris_crash) msg

%foreign (libxla "isnull")
prim__isNullPtr : AnyPtr -> Int

export
isNullPtr : AnyPtr -> Bool
isNullPtr ptr = cIntToBool $ prim__isNullPtr ptr

export
boolToCInt : Bool -> Int
boolToCInt True = 1
boolToCInt False = 0

public export
data IntArray : Type where
  MkIntArray : GCPtr Int -> IntArray

%foreign (libxla "sizeof_int")
sizeofInt : Int

%foreign (libxla "set_array_int")
prim__setArrayInt : Ptr Int -> Int -> Int -> PrimIO ()

export
mkIntArray : (HasIO io, Cast a Int) => List a -> io IntArray
mkIntArray xs = do
  ptr <- malloc (cast (length xs) * sizeofInt)
  let ptr = prim__castPtr ptr
  traverse_ (\(idx, x) => primIO $ prim__setArrayInt ptr (cast idx) (cast x)) (enumerate xs)
  ptr <- onCollect ptr (free . prim__forgetPtr)
  pure (MkIntArray ptr)

export
%foreign (libxla "sizeof_ptr")
sizeofPtr : Int

export
%foreign (libxla "set_array_ptr")
prim__setArrayPtr : AnyPtr -> Int -> AnyPtr -> PrimIO ()
