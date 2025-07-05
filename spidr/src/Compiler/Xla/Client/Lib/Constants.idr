{--
Copyright (C) 2022  Joel Berkeley

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
||| For internal spidr use only.
module Compiler.Xla.Client.Lib.Constants

import Compiler.FFI
import Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.XlaData

%foreign (libxla "MinValue")
prim__minValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
minValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
minValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__minValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "MinFiniteValue")
prim__minFiniteValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
minFiniteValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
minFiniteValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__minFiniteValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "MaxValue")
prim__maxValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
maxValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
maxValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__maxValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "MaxFiniteValue")
prim__maxFiniteValue : GCAnyPtr -> Int -> PrimIO AnyPtr

export
maxFiniteValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
maxFiniteValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__maxFiniteValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)
