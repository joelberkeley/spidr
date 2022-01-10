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
module Unit.TestXLA

import Data.Vect
import System.FFI

import XLA.Client.ClientLibrary
import XLA.Client.LocalClient
import XLA.Client.XlaBuilder
import XLA.Client.XlaComputation
import XLA.FFI
import XLA.Literal
import XLA.Shape
import XLA.ShapeUtil
import XLA.XlaData
import Types

import Util'

export
test_parameter_addition : IO ()
test_parameter_addition = do
    -- todo free and delete

    -- function definition
    builder <- primIO (prim__mkXlaBuilder "")
    c_shape <- mkIntArray [2, 3]
    xla_shape <- primIO (prim__mkShape (cast $ primitiveType {dtype=Int}) c_shape 2)
    p0 <- collectXlaOp (parameter builder 0 xla_shape "")
    p1 <- collectXlaOp (parameter builder 1 xla_shape "")
    delete xla_shape
    free c_shape
    _ <- primIO (prim__add p0 p1)

    -- user code
    p0_lit <- mkLiteral {shape=[2, 3]} {dtype=Int} [[0, 1, 2], [3, 4, 5]]
    p1_lit <- mkLiteral {shape=[2, 3]} {dtype=Int} [[1, 1, 1], [-1, -1, -1]]
    -- in `eval`
    client <- primIO prim__localClientOrDie
    gd0 <- primIO (prim__transferToServer client p0_lit)
    gd1 <- primIO (prim__transferToServer client p1_lit)
    -- not convinced this is right yet. Could just have a sizeof_GlobalData function which is less
    -- useful but safe
    gd_arr <- malloc (2 * sizeof_voidPtr)
    primIO (prim__setArrayPtr gd_arr 0 gd0)
    primIO (prim__setArrayPtr gd_arr 1 gd1)
    lit <- primIO $ prim__executeAndTransfer client (build builder) gd_arr 2

    -- user code
    assert (toArray {shape=[2, 3]} {dtype=Int} lit == [[1, 2, 3], [2, 3, 4]])
