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
||| For internal spidr use only.
module Compiler.Xla.HLO.Builder.Lib.Matrix

import Compiler.FFI
import Compiler.Xla.HLO.Builder.XlaBuilder
import Compiler.Xla.XlaData

%foreign (libxla "IdentityMatrix")
prim__identityMatrix : GCAnyPtr -> Int -> Int -> Int -> PrimIO AnyPtr

export
identityMatrix : HasIO io => Primitive dtype => XlaBuilder -> Nat -> Nat -> io XlaOp
identityMatrix (MkXlaBuilder builder) m n = do
  opPtr <- primIO $ prim__identityMatrix builder (xlaIdentifier {dtype}) (cast m) (cast n)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "GetMatrixDiagonal")
prim__getMatrixDiagonal : GCAnyPtr -> PrimIO AnyPtr

export
getMatrixDiagonal : HasIO io => XlaOp -> io XlaOp
getMatrixDiagonal (MkXlaOp x) = do
  opPtr <- primIO $ prim__getMatrixDiagonal x
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Triangle")
prim__triangle : GCAnyPtr -> Int -> PrimIO AnyPtr

export
triangle : HasIO io => XlaOp -> Bool -> io XlaOp
triangle (MkXlaOp x) lower = do
  opPtr <- primIO $ prim__triangle x (boolToCInt lower)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)
