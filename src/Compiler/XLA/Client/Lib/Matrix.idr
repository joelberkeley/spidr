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
module Compiler.XLA.Client.Lib.Matrix

import Data.Hashable
import Control.Monad.State
import System.FFI
import Data.SortedMap

import Compiler.FFI
import Compiler.Graph
import Compiler.XLA.Client.XlaBuilder
import Primitive

%foreign (libxla "IdentityMatrix")
prim__identityMatrixImpl : GCAnyPtr -> Int -> Int -> Int -> PrimIO AnyPtr

export
prim__identityMatrix : Primitive dtype => Int -> Int -> Graph -> XlaOpFactory
prim__identityMatrix m n graph = do
  MkXlaBuilder ptr _ <- get
  op <- primIO $ prim__identityMatrixImpl ptr (xlaIdentifier {dtype}) m n
  onCollectAny op XlaOp.delete

export
%foreign (libxla "GetMatrixDiagonal")
prim__getMatrixDiagonal : GCAnyPtr -> PrimIO AnyPtr

export
%foreign (libxla "Triangle")
prim__triangle : GCAnyPtr -> Int -> PrimIO AnyPtr
