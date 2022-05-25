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

||| A `Graph` is a tree representation of the computational graph of a tensor value.
||| It is not intended to represent the actual computation graph executed at runtime, rather its
||| primary purpose is to avoid duplication within the graph.
|||
||| For any two `Graph` objects gx and gy, which correspond to computational graphs computing
||| tensors x and y respectively, if gx is equal to gy, then x is equal to y, though the
||| computations used to compute x and y may be different.
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
Prelude.Eq Graph where
  FromLiteral {dtype} shape hash == FromLiteral {dtype=dtype'} shape' hash' =
    (typeString {dtype}, shape, hash) == (typeString {dtype=dtype'}, shape', hash')
  Parameter {dtype} shape position == Parameter {dtype=dtype'} shape' position' =
    (typeString {dtype}, shape, position) == (typeString {dtype=dtype'}, shape', position')
  Reshape to x == Reshape to' x' = to == to' && x == x'
  Slice axis from to x == Slice axis' from' to' x' = (axis, from, to, x) == (axis', from', to', x')
  Concat axis x y == Concat axis' x' y' = (axis, x, y) == (axis', x', y')
  Diag x == Diag x' = x == x'
  Triangle lower x == Triangle lower' x' = (lower, x) == (lower', x')
  Transpose x == Transpose x' = x == x'
  Identity {dtype} n == Identity {dtype=dtype'} n' =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  Broadcast to x == Broadcast to' x' = (to, x) == (to', x')
  Map f xs == Map f' xs' = (f, xs) == (f', xs')
  Reduce monoid axis x == Reduce monoid' axis' x' = (monoid, axis, x) == (monoid', axis', x')
  ElementwiseBinary name x y == ElementwiseBinary name' x' y' = (name, x, y) == (name', x', y')
  ElementwiseUnary name x == ElementwiseUnary name' x' = (name, x) == (name', x')
  Select pred t f == Select pred' t' f' = (pred, t, f) == (pred', t', f')
  Cond pred fTrue true fFalse false == Cond pred' fTrue' true' fFalse' false' =
    (pred, fTrue, true, fFalse, false) == (pred', fTrue', true', fFalse', false')
  Dot x y == Dot x' y' = (x, y) == (x', y')
  Cholesky x == Cholesky x' = x == x'
  TriangularSolve lower x y == TriangularSolve lower' x' y' = (lower, x, y) == (lower', x', y')
  _ == _ = False

export covering
Hashable Graph where
  hashWithSalt salt (FromLiteral {dtype} hash shape) =
    salt `hashWithSalt` "FromLiteral" `hashWithSalt`  (typeString {dtype}, shape, hash)
  hashWithSalt salt (Parameter {dtype} shape position) =
    salt `hashWithSalt` "Parameter" `hashWithSalt` (typeString {dtype}, shape, position)
  hashWithSalt salt (Reshape to x) = salt `hashWithSalt` "Reshape" `hashWithSalt` (to, x)
  hashWithSalt salt (Slice axis from to x) =
    salt `hashWithSalt` "Slice" `hashWithSalt` (axis, from, to)
  hashWithSalt salt (Concat axis x y) = salt `hashWithSalt` "Concat" `hashWithSalt` (axis, x, y)
  hashWithSalt salt (Diag x) = salt `hashWithSalt` "Diag" `hashWithSalt` x
  hashWithSalt salt (Triangle lower x) = salt `hashWithSalt` "Triangle" `hashWithSalt` (lower, x)
  hashWithSalt salt (Transpose x) = salt `hashWithSalt` "Transpose" `hashWithSalt` x
  hashWithSalt salt (Identity {dtype} n) =
    salt `hashWithSalt` "Identity" `hashWithSalt` (typeString {dtype}, n)
  hashWithSalt salt (Broadcast to x) = salt `hashWithSalt` "Broadcast" `hashWithSalt` (to, x)
  hashWithSalt salt (Map f xs) = salt `hashWithSalt` "Map" `hashWithSalt` (f, xs)
  hashWithSalt salt (Reduce monoid axis x) =
    salt `hashWithSalt` "Reduce" `hashWithSalt` (monoid, axis, x)
  hashWithSalt salt (ElementwiseBinary name x y) = hashWithSalt salt (name, x, y)
  hashWithSalt salt (ElementwiseUnary name x) = hashWithSalt salt (name, x)
  hashWithSalt salt (Select pred f t) = salt `hashWithSalt` "Select" `hashWithSalt` (pred, t, f)
  hashWithSalt salt (Cond pred fTrue true fFalse false) =
    salt `hashWithSalt` "Cond" `hashWithSalt` (pred, fTrue, true, fFalse, false)
  hashWithSalt salt (Dot x y) = salt `hashWithSalt` "Dot" `hashWithSalt` (x, y)
  hashWithSalt salt (Cholesky x) = salt `hashWithSalt` "Cholesky" `hashWithSalt` x
  hashWithSalt salt (TriangularSolve lower x y) =
    salt `hashWithSalt` "TriangularSolve" `hashWithSalt` (lower, x, y)
