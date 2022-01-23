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
||| This module defines supported primitive backend types and their interaction with Idris.
module Primitive

import public XLA.XlaData
import XLA.Literal

||| A `PrimitiveRW a b` constitutes proof that we can read and write between a backend primitive
||| type `dtype` and an Idris type `idr`.
export
interface LiteralPrimitiveRW dtype idr => PrimitiveRW dtype idr | dtype where

export PrimitiveRW PRED Bool where
export PrimitiveRW S32 Int where
export PrimitiveRW U32 Nat where
export PrimitiveRW F64 Double where
