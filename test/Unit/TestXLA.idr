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

import Utils

test_parameter_addition : IO ()
test_parameter_addition = do
  builder <- prim__mkXlaBuilder ""
  xla_shape <- mkShape {dtype=S32} [2, 3]
  p0 <- onCollectAny (parameter builder 0 xla_shape "") XlaOp.delete
  p1 <- onCollectAny (parameter builder 1 xla_shape "") XlaOp.delete
  sum <- primIO (prim__add p0 p1)
  _ <- onCollectAny sum XlaOp.delete
  computation <- prim__build builder

  p0_lit <- mkLiteral {shape=[2, 3]} {dtype=S32} {ty=Int} [[0, 1, 2], [3, 4, 5]]
  p1_lit <- mkLiteral {shape=[2, 3]} {dtype=S32} {ty=Int} [[1, 1, 1], [-1, -1, -1]]

  client <- primIO prim__localClientOrDie
  gd0 <- prim__transferToServer client p0_lit
  gd1 <- prim__transferToServer client p1_lit
  gd_arr <- malloc (2 * sizeof_voidPtr)
  primIO (prim__setArrayPtr gd_arr 0 gd0)
  primIO (prim__setArrayPtr gd_arr 1 gd1)
  lit <- prim__executeAndTransfer client computation gd_arr 2
  free gd_arr

  let sum = toArray {shape=[2, 3]} {dtype=S32} {ty=Int} lit
  assert "array addition using Parameter" (sum == [[1, 2, 3], [2, 3, 4]])

export
test : IO ()
test = do
  test_parameter_addition
