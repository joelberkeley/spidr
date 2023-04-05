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
module Compiler.Xla.TensorFlow.Compiler.Xla.XlaData

export
interface Primitive dtype where
  xlaIdentifier : Int

export data PRED : Type where

export
Primitive PRED where
  xlaIdentifier = 1

export data S32 : Type where

export
Primitive S32 where
  xlaIdentifier = 4

export data S64 : Type where

export
Primitive S64 where
  xlaIdentifier = 5

export data U32 : Type where

export
Primitive U32 where
  xlaIdentifier = 8

export data U64 : Type where

export
Primitive U64 where
  xlaIdentifier = 9

export data F32 : Type where

export
Primitive F32 where
  xlaIdentifier = 11

export data F64 : Type where

export
Primitive F64 where
  xlaIdentifier = 12
