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

public export
data PrimitiveType = PRED | S32 | S64 | U32 | U64 | F32 | F64

export
Cast PrimitiveType Int where
    cast PRED = 1
    cast S32 = 4
    cast S64 = 5
    cast U32 = 8
    cast U64 = 9
    cast F32 = 11
    cast F64 = 12

public export
interface XLAPrimitive dtype where
    primitiveType : PrimitiveType
    set : GCAnyPtr -> Ptr Int -> dtype -> PrimIO ()
    get : GCAnyPtr -> Ptr Int -> dtype
