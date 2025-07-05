{--
Copyright (C) 2021  Joel Berkeley

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
module Compiler.Xla.XlaData

import Compiler.FFI

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

namespace Xla
  public export
  data DotDimensionNumbers : Type where
    MkDotDimensionNumbers : GCAnyPtr -> DotDimensionNumbers

namespace DotDimensionNumbers
  %foreign (libxla "DotDimensionNumbers_delete")
  prim__dotDimensionNumbersDelete : AnyPtr -> PrimIO ()

  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . prim__dotDimensionNumbersDelete

%foreign (libxla "DotDimensionNumbers_new")
prim__dotDimensionNumbersNew : PrimIO AnyPtr

export
allocDotDimensionNumbers : HasIO io => io DotDimensionNumbers
allocDotDimensionNumbers = do
  ptr <- primIO prim__dotDimensionNumbersNew
  ptr <- onCollectAny ptr DotDimensionNumbers.delete
  pure (MkDotDimensionNumbers ptr)

%foreign (libxla "DotDimensionNumbers_add_lhs_contracting_dimensions")
prim__addLhsContractingDimensions : GCAnyPtr -> Int -> PrimIO ()

export
addLhsContractingDimensions : HasIO io => DotDimensionNumbers -> Nat -> io ()
addLhsContractingDimensions (MkDotDimensionNumbers dimensionNumbers) n =
  primIO $ prim__addLhsContractingDimensions dimensionNumbers (cast n)

%foreign (libxla "DotDimensionNumbers_add_rhs_contracting_dimensions")
prim__addRhsContractingDimensions : GCAnyPtr -> Int -> PrimIO ()

export
addRhsContractingDimensions : HasIO io => DotDimensionNumbers -> Nat -> io ()
addRhsContractingDimensions (MkDotDimensionNumbers dimensionNumbers) n =
  primIO $ prim__addRhsContractingDimensions dimensionNumbers (cast n)

%foreign (libxla "DotDimensionNumbers_add_lhs_batch_dimensions")
prim__addLhsBatchDimensions : GCAnyPtr -> Int -> PrimIO ()

export
addLhsBatchDimensions : HasIO io => DotDimensionNumbers -> Nat -> io ()
addLhsBatchDimensions (MkDotDimensionNumbers dimensionNumbers) n =
  primIO $ prim__addLhsBatchDimensions dimensionNumbers (cast n)

%foreign (libxla "DotDimensionNumbers_add_rhs_batch_dimensions")
prim__addRhsBatchDimensions : GCAnyPtr -> Int -> PrimIO ()

export
addRhsBatchDimensions : HasIO io => DotDimensionNumbers -> Nat -> io ()
addRhsBatchDimensions (MkDotDimensionNumbers dimensionNumbers) n =
  primIO $ prim__addRhsBatchDimensions dimensionNumbers (cast n)
