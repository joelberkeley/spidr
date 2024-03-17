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
module Compiler.Xla.Client.Lib.Matrix

import Compiler.FFI
import Compiler.Xla.Client.XlaBuilder
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
