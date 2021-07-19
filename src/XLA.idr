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
Scalar = Struct "cScalar" [("x", Double)]

%foreign (libxla "cScalar_new")
export
mkScalar : Double -> Scalar

%foreign (libxla "cScalar_del")
prim__delScalar : Scalar -> PrimIO ()

export
delScalar : Scalar -> IO ()
delScalar = primIO . prim__delScalar

%foreign (libxla "cScalar_add")
export
add : Scalar -> Scalar -> Scalar

%foreign (libxla "cScalar_toDouble")
toDouble : Scalar -> Double

export
Cast Scalar Double where
  cast = toDouble

||| Scalar data types supported by XLA.
public export
data ArchType = BOOL | U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 | F16 | F32 | F64
