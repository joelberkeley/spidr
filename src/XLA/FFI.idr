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
import System.FFI

import Types
import Util

export
enumerate : Vect n ty -> Vect n (Nat, ty)
enumerate xs = rewrite sym $ lengthCorrect xs in
    zip (range (length xs)) (rewrite lengthCorrect xs in xs)

export
free : Ptr t -> IO ()
free = System.FFI.free . prim__forgetPtr

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libc_xla_extension"

%foreign (libxla "sizeof_int")
sizeof_int : Int

{-
 -
 - Array shape
 -
 -}

%foreign (libxla "set_array_int")
prim__setArrayInt : Ptr Int -> Int -> Int -> PrimIO ()

export
mkIntArray : Cast ty Int => Vect n ty -> IO (Ptr Int)
mkIntArray xs = do
    ptr <- malloc (cast (length xs) * sizeof_int)
    let ptr = prim__castPtr ptr
    traverse_ (\(idx, x) => primIO $ prim__setArrayInt ptr (cast idx) (cast x)) (enumerate xs)
    pure ptr
