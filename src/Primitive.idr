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

%hide Prelude.Num
%hide Prelude.Neg
%hide Prelude.Abs
%hide Prelude.Fractional

export interface Num dtype where a_ : Nat
export interface Num dtype => Neg dtype where
export interface Num dtype => Abs dtype where
export interface Num dtype => Fractional dtype where

export Num U32 where a_ = 0
export Num U64 where a_ = 0
export Num S32 where a_ = 0
export Num S64 where a_ = 0
export Num F32 where a_ = 0
export Num F64 where a_ = 0

export Neg S32 where
export Neg S64 where
export Neg F32 where
export Neg F64 where

export Abs S32 where
export Abs S64 where
export Abs F32 where
export Abs F64 where

export Fractional F32 where
export Fractional F64 where

%hide Prelude.Eq
%hide Prelude.Ord

export interface Eq dtype where b_ : Nat
export interface Eq dtype => Ord dtype where

export Primitive dtype => Eq dtype where b_ = 0

export Ord U32 where
export Ord U64 where
export Ord S32 where
export Ord S64 where
export Ord F32 where
export Ord F64 where

||| A `PrimitiveRW a b` constitutes proof that we can read and write between a backend primitive
||| type `dtype` and an Idris type `idr`.
export
interface LiteralPrimitiveRW dtype idr => PrimitiveRW dtype idr | dtype where

export PrimitiveRW PRED Bool where
export PrimitiveRW S32 Int where
export PrimitiveRW U32 Nat where
export PrimitiveRW F64 Double where
