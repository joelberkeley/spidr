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
module Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Constants

import Compiler.Xla.Prim.TensorFlow.Compiler.Xla.Client.Lib.Constants
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData

export
minValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
minValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__minValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
minFiniteValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
minFiniteValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__minFiniteValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
maxValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
maxValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__maxValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
maxFiniteValue : (HasIO io, Primitive dtype) => XlaBuilder -> io XlaOp
maxFiniteValue (MkXlaBuilder builder) = do
  opPtr <- primIO $ prim__maxFiniteValue builder (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)
