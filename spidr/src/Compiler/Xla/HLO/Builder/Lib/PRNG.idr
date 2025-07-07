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
module Compiler.Xla.HLO.Builder.Lib.PRNG

import Compiler.FFI
import Compiler.Xla.HLO.Builder.XlaBuilder
import Compiler.Xla.Shape

public export
data BitGenerator = ThreeFry | Philox

Cast BitGenerator Int where
  cast ThreeFry = 0
  cast Philox = 1

public export
record RngOutput where
  constructor MkRngOutput
  value : XlaOp
  state : XlaOp

PrimRngOutput : Type
PrimRngOutput = Struct "PrimRngOutput" [("value", AnyPtr), ("state", AnyPtr)]

%foreign (libxla "delete_RngOutput")
prim__delete : PrimRngOutput -> PrimIO ()

%foreign (libxla "UniformFloatingPointDistribution")
prim__uniformFloatingPointDistribution:
  GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO PrimRngOutput

export
uniformFloatingPointDistribution :
  HasIO io => XlaOp -> XlaOp -> BitGenerator -> XlaOp -> XlaOp -> Shape -> io RngOutput
uniformFloatingPointDistribution
  (MkXlaOp key)
  (MkXlaOp initialState)
  bitGenerator
  (MkXlaOp minval)
  (MkXlaOp maxval)
  (MkShape shape) = do
    rngOutput <- primIO $ prim__uniformFloatingPointDistribution
      key initialState (cast bitGenerator) minval maxval shape
    let value = getField rngOutput "value"
        state = getField rngOutput "state"
    primIO $ prim__delete rngOutput
    value <- onCollectAny value XlaOp.delete
    state <- onCollectAny state XlaOp.delete
    pure (MkRngOutput {value = MkXlaOp value} {state = MkXlaOp state})

%foreign (libxla "NormalFloatingPointDistribution")
prim__normalFloatingPointDistribution:
  GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr -> PrimIO PrimRngOutput

export
normalFloatingPointDistribution :
  HasIO io => XlaOp -> XlaOp -> BitGenerator -> Shape -> io RngOutput
normalFloatingPointDistribution
  (MkXlaOp key) (MkXlaOp initialState) bitGenerator (MkShape shape) = do
    rngOutput <- primIO $
      prim__normalFloatingPointDistribution key initialState (cast bitGenerator) shape
    let value = getField rngOutput "value"
        state = getField rngOutput "state"
    primIO $ prim__delete rngOutput
    value <- onCollectAny value XlaOp.delete
    state <- onCollectAny state XlaOp.delete
    pure (MkRngOutput {value = MkXlaOp value} {state = MkXlaOp state})
