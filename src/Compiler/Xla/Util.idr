{--
Copyright 2021 Joel Berkeley

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
module Compiler.Xla.Util

import System.FFI

import Compiler.Xla.Prim.Util
import Util.List

export
cIntToBool : Int -> Bool
cIntToBool 0 = False
cIntToBool 1 = True
cIntToBool x =
  let msg = "Internal error: expected 0 or 1 from XLA C API for boolean conversion, got " ++ show x
  in (assert_total idris_crash) msg

export
boolToCInt : Bool -> Int
boolToCInt True = 1
boolToCInt False = 0

public export
data IntArray : Type where
  MkIntArray : GCPtr Int -> IntArray

export
mkIntArray : (HasIO io, Cast a Int) => List a -> io IntArray
mkIntArray xs = do
  ptr <- malloc (cast (length xs) * sizeofInt)
  let ptr = prim__castPtr ptr
  traverse_ (\(idx, x) => primIO $ prim__setArrayInt ptr (cast idx) (cast x)) (enumerate xs)
  ptr <- onCollect ptr (free . prim__forgetPtr)
  pure (MkIntArray ptr)
