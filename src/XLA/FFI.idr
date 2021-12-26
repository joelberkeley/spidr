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
module XLA.FFI

import Data.Vect
import Types
import Util

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

{-
 -
 - Array shape
 -
 -}

%foreign (libxla "alloc_int_array")
prim__allocIntArray : Int -> PrimIO (Ptr Int)

%foreign (libxla "free_int_array")
prim__freeIntArray : Ptr Int -> PrimIO ()

%foreign (libxla "set_array_int")
prim__setArrayInt : Ptr Int -> Int -> Int -> PrimIO ()

export
mkIntArray : {rank : _} -> Shape {rank} -> IO (Ptr Int)
mkIntArray xs = do
    ptr <- primIO $ prim__allocIntArray (cast rank)
    foldl (writeElem ptr) (pure ()) (zip (range rank) xs)
    pure ptr where
        writeElem : Ptr Int -> IO () -> (Nat, Nat) -> IO ()
        writeElem ptr prev_io (idx, x) = do
            prev_io
            primIO $ prim__setArrayInt ptr (cast idx) (cast x)

export
free : Ptr Int -> IO ()
free = primIO . prim__freeIntArray
