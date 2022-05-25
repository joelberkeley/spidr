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
module Compiler.Graph

import Primitive
import Data.Hashable
import Data.Stream
import Types
import Util

||| A `Graph` represents a computational graph used to compute a tensor value. It is defined as
||| follows: For any two `Graph`s gx and gy that compute tensors x and y respectively, if gx is
||| equal to gy, then the values of x and y are equal.
|||
||| It is primarily used for memoization in constructing the computation graph.
public export
data Graph : Type where
  FromLiteral : Primitive dtype => Shape -> (hash : Bits64) -> Graph
  Parameter : Primitive dtype => Shape -> Nat -> Graph
  Reshape : Shape -> Graph -> Graph
  Slice : Nat -> Nat -> Nat -> Graph -> Graph
  Concat : Nat -> Graph -> Graph -> Graph
  Diag : Graph -> Graph
  Triangle : (lower : Bool) -> Graph -> Graph
  Transpose : Graph -> Graph
  Identity : Primitive dtype => Nat -> Graph
  Broadcast : Shape -> Graph -> Graph
  Map : Graph -> List Graph -> Graph
  Reduce : Graph -> Nat -> Graph -> Graph
  ElementwiseBinary : (name : String) -> Graph -> Graph -> Graph
  ElementwiseUnary : (name : String) -> Graph -> Graph
  Select : Graph -> Graph -> Graph -> Graph
  Cond : Graph -> Graph -> Graph -> Graph -> Graph -> Graph
  Dot : Graph -> Graph -> Graph
  Cholesky : Graph -> Graph
  TriangularSolve : (lower : Bool) -> Graph -> Graph -> Graph

export covering
Hashable Graph where
  hashWithSalt salt (FromLiteral {dtype} hash shape) =
    salt `hashWithSalt` ("FromLiteral", typeString {dtype}, shape, hash)
  hashWithSalt salt (Parameter {dtype} shape position) =
    salt `hashWithSalt` ("Parameter", typeString {dtype}, shape, position)
  hashWithSalt salt (Reshape to x) = salt `hashWithSalt` ("Reshape", to, x)
  hashWithSalt salt (Slice axis from to x) = salt `hashWithSalt` ("Slice", axis, from, to)
  hashWithSalt salt (Concat axis x y) = salt `hashWithSalt` ("Concat", axis, x, y)
  hashWithSalt salt (Diag x) = salt `hashWithSalt` ("Diag", x)
  hashWithSalt salt (Triangle lower x) = salt `hashWithSalt` ("Triangle", lower, x)
  hashWithSalt salt (Transpose x) = salt `hashWithSalt` ("Transpose", x)
  hashWithSalt salt (Identity {dtype} n) = salt `hashWithSalt` ("Identity", typeString {dtype}, n)
  hashWithSalt salt (Broadcast to x) = salt `hashWithSalt` ("Broadcast", to, x)
  hashWithSalt salt (Map f xs) = salt `hashWithSalt` ("Map", f, xs)
  hashWithSalt salt (Reduce monoid axis x) = salt `hashWithSalt` ("Reduce", monoid, axis, x)
  hashWithSalt salt (ElementwiseBinary name x y) = hashWithSalt salt (name, x, y)
  hashWithSalt salt (ElementwiseUnary name x) = hashWithSalt salt (name, x)
  hashWithSalt salt (Select pred f t) = salt `hashWithSalt` ("Select", pred, f, t)
  hashWithSalt salt (Cond pred fTrue true fFalse false) =
    salt `hashWithSalt` "Cond" `hashWithSalt` (pred, fTrue, true, fFalse, false)
  hashWithSalt salt (Dot x y) = salt `hashWithSalt` ("Dot", x, y)
  hashWithSalt salt (Cholesky x) = salt `hashWithSalt` ("Cholesky", x)
  hashWithSalt salt (TriangularSolve lower x y) =
    salt `hashWithSalt` ("TriangularSolve", lower, x, y)
