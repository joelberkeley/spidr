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

export
Bignum : Type
Bignum = Struct "c__Bignum" []

%foreign (libxla "c__Bignum_Bignum")
export
mkBignum : Bignum

export
delete : Bignum -> IO ()
delete = primIO . prim__delete where
           %foreign (libxla "c__Bignum_del")
           prim__delete : Bignum -> PrimIO ()

export
assign : Bignum -> Nat -> IO ()
assign b x = primIO $ prim__assign b (cast x) where
               %foreign (libxla "c__Bignum_AssignUInt64")
               prim__assign : Bignum -> Int -> PrimIO ()


export
add : Bignum -> Bignum -> IO ()
add x y = primIO $ prim__c__Bignum_AddBignum x y where
            %foreign (libxla "c__Bignum_AddBignum")
            prim__c__Bignum_AddBignum : Bignum -> Bignum -> PrimIO ()

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
                  _ => LT  -- todo crash

||| Scalar data types supported by XLA.
public export
data ArchType = BOOL | U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 | F16 | F32 | F64
