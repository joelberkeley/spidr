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
||| This module contains the Idris API to XLA.
module XLA

import System.FFI

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

xla_crash : Show a => a -> b
xla_crash x = (assert_total idris_crash) $ "Fatal: XLA C API produced unexpected value " ++ show x

export
Bignum : Type
Bignum = Struct "c__Bignum" []

%foreign (libxla "c__Bignum_Bignum")
export
mkBignum : Bignum

%foreign (libxla "c__Bignum_del")
prim__delete : Bignum -> PrimIO ()

export
delete : Bignum -> IO ()
delete = primIO . prim__delete

%foreign (libxla "c__Bignum_AssignUInt64")
prim__assign : Bignum -> Int -> PrimIO ()

export
assign : Bignum -> Nat -> IO ()
assign b x = primIO $ prim__assign b (cast x)

%foreign (libxla "c__Bignum_AddBignum")
prim__c__Bignum_AddBignum : Bignum -> Bignum -> PrimIO ()

export
add : Bignum -> Bignum -> IO ()
add x y = primIO $ prim__c__Bignum_AddBignum x y

%foreign (libxla "c__Bignum_Compare")
prim__compare : Bignum -> Bignum -> Int

export
Eq Bignum where
  x == y = if prim__compare x y == 0 then True else False

export
Ord Bignum where
  compare x y = case prim__compare x y of
                  -1 => LT
                  0 => EQ
                  1 => GT
                  x => xla_crash x

||| Scalar data types supported by XLA.
public export
data ArchType = BOOL | U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 | F16 | F32 | F64
