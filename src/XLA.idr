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
||| This module contains (will contain) the Idris API to XLA.
module XLA

import System.FFI

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

export
Scalar : Type
Scalar = Struct "cBignum" [("x", Int)]

%foreign (libxla "cBignum_new")
export
mkScalar : Int -> Scalar

%foreign (libxla "cBignum_del")
prim__delScalar : Scalar -> PrimIO ()

export
delScalar : Scalar -> IO ()
delScalar = primIO . prim__delScalar

%foreign (libxla "cBignum_add")
export
add : Scalar -> Scalar -> Scalar

%foreign (libxla "cBignum_compare")
prim__compare : Scalar -> Scalar -> Int

export
Eq Scalar where
  x == y = if prim__compare x y == 0 then True else False

export
Ord Scalar where
  compare x y = case prim__compare x y of
                  -1 => LT
                  0 => EQ
                  1 => GT
                  _ => LT

||| Scalar data types supported by XLA.
public export
data ArchType = BOOL | U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 | F16 | F32 | F64
