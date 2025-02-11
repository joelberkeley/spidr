{--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
||| Supported backend primitive types, and their relation to Idris primitives.
|||
||| The module contains a number of interfaces (`Primitive.Num`, `Primitive.Eq` etc.). These
||| indicate what operations can be performed on primitive data in the backend. They are entirely
||| distinct from the Idris interfaces `Prelude.Num` etc. but carry largely the same meaning.
||| For example, primitive types satsifying `Primitive.Ord` have a notion of ordering.
module Primitive

import Compiler.LiteralRW
import public Compiler.Xla.XlaData

%hide Prelude.Num
%hide Prelude.Neg
%hide Prelude.Abs
%hide Prelude.Integral
%hide Prelude.Fractional

export interface Primitive dtype => Num dtype where
export interface Num dtype => Neg dtype where
export interface Num dtype => Abs dtype where
export interface Num dtype => Integral dtype where
export interface Num dtype => Fractional dtype where

export Num U32 where
export Num U64 where
export Num S32 where
export Num S64 where
export Num F32 where
export Num F64 where

export Neg S32 where
export Neg S64 where
export Neg F32 where
export Neg F64 where

export Abs S32 where
export Abs S64 where
export Abs F32 where
export Abs F64 where

export Integral U32 where
export Integral U64 where
export Integral S32 where
export Integral S64 where

export Fractional F32 where
export Fractional F64 where

%hide Prelude.Eq
%hide Prelude.Ord

export interface Primitive dtype => Eq dtype where
export interface Eq dtype => Ord dtype where

export Eq PRED where
export Eq U32 where
export Eq U64 where
export Eq S32 where
export Eq S64 where
export Eq F32 where
export Eq F64 where

export Ord U32 where
export Ord U64 where
export Ord S32 where
export Ord S64 where
export Ord F32 where
export Ord F64 where

||| A `PrimitiveRW dtype idr` means that values of type `idr` can be used to construct backend
||| data with data type `dtype`.
export
interface LiteralRW dtype idr => PrimitiveRW dtype idr | dtype where

export PrimitiveRW PRED Bool where
export PrimitiveRW S32 Int32 where
export PrimitiveRW U32 Nat where
export PrimitiveRW U64 Nat where
export PrimitiveRW F64 Double where
