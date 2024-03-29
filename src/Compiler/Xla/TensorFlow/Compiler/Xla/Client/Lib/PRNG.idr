{--
Copyright 2022 Joel Berkeley

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
module Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.PRNG

import System.FFI

import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape

public export
data BitGenerator = ThreeFry | Philox

Cast BitGenerator Int where
  cast ThreeFry = 0
  cast Philox = 1

%hide Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.PRNG.RngOutput

public export
record RngOutput where
  constructor MkRngOutput
  value : XlaOp
  state : XlaOp

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
