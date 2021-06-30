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
module Poplar

import System.FFI

libpoplar : String -> String
libpoplar fname = "C:" ++ fname ++ ",libpoplar"

export
Scalar : Type
Scalar = Struct "cScalar" [("x", Double)]

%foreign (libpoplar "cScalar_new")
export
mkScalar : Double -> Scalar

%foreign (libpoplar "cScalar_del")
prim__delScalar : Scalar -> PrimIO ()

export
delScalar : Scalar -> IO ()
delScalar = primIO . prim__delScalar

%foreign (libpoplar "cScalar_add")
export
add : Scalar -> Scalar -> Scalar

%foreign (libpoplar "cScalar_toDouble")
toDouble : Scalar -> Double

export
Cast Scalar Double where
  cast = toDouble

||| Scalar data types supported by Poplar.
public export
data ArchType = BOOL | U8 | I8 | U16 | I16 | U32 | I32 | U64 | I64 | F16 | F32
