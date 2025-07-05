{--
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
