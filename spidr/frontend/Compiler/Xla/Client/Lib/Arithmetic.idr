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
||| For internal spidr use only.
module Compiler.Xla.Client.Lib.Arithmetic

import Compiler.FFI
import Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.XlaData

%foreign (libxla "ArgMax")
prim__argMax : GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

export
argMax : (HasIO io, Primitive outputType) => XlaOp -> Nat -> io XlaOp
argMax (MkXlaOp input) axis = do
  opPtr <- primIO $ prim__argMax input (xlaIdentifier {dtype = outputType}) (cast axis)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "ArgMin")
prim__argMin : GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

export
argMin : (HasIO io, Primitive outputType) => XlaOp -> Nat -> io XlaOp
argMin (MkXlaOp input) axis = do
  opPtr <- primIO $ prim__argMin input (xlaIdentifier {dtype = outputType}) (cast axis)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)
