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
module Compiler.FFI

import public System.FFI
import Util

public export
libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libc_xla"

public export
data CppString = MkCppString GCAnyPtr

%foreign (libxla "string_delete")
prim__stringDelete : AnyPtr -> PrimIO ()

namespace CppString
  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . prim__stringDelete

%foreign (libxla "string_c_str")
prim__stringCStr : GCAnyPtr -> PrimIO String

export
cstr : HasIO io => CppString -> io String
cstr (MkCppString str) = primIO $ prim__stringCStr str

%foreign (libxla "string_size")
prim__stringSize : GCAnyPtr -> Int

export
size : CppString -> Int
size (MkCppString str) = prim__stringSize str

export
%foreign (libxla "sizeof_int")
sizeofInt : Int

export
%foreign (libxla "sizeof_ptr")
sizeofPtr : Int

export
%foreign (libxla "index")
prim__index : Int -> AnyPtr -> AnyPtr

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
