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
module XLA.XlaData

||| Primitive data types supported by the backend.
public export
data Primitive = PRED | S32 | S64 | U32 | U64 | F32 | F64

export
Cast Primitive Int where
    cast PRED = 1
    cast S32 = 4
    cast S64 = 5
    cast U32 = 8
    cast U64 = 9
    cast F32 = 11
    cast F64 = 12

||| A `ScalarLike` is an Idris type for which there is a correspnding backend primitive type.
export
interface ScalarLike dtype where
  primitiveType : Primitive

export
ScalarLike Double where
  primitiveType = F64

export
ScalarLike Int where
  primitiveType = S32

export
ScalarLike Integer where
  primitiveType = S64

export
ScalarLike Nat where
  primitiveType = U64

export
ScalarLike Bool where
  primitiveType = PRED
