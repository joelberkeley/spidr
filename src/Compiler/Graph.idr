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

import Data.Hashable

import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Types
import Util.Hashable

||| A `Graph` represents a computational graph used to compute a tensor value. It is defined by
||| the following proprty: For any two `Graph`s gx and gy that compute tensors x and y respectively,
||| if gx is identical to gy, then the values of x and y are equal.
|||
||| It is primarily used for memoization in constructing the computation graph.
public export
data Graph : Type where
  FromLiteral : Primitive dtype => Shape -> (hash : Bits64) -> Graph
  Parameter : Primitive dtype => Shape -> Nat -> Graph
  MinFiniteValue : Primitive dtype => Graph
  MaxFiniteValue : Primitive dtype => Graph
  ConvertElementType : Primitive dtype => Graph -> Graph
  GetTupleElement : Graph -> Nat -> Graph
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
  Sort : List Graph -> Graph -> Nat -> Bool -> Graph
  Reverse : List Nat -> Graph -> Graph
  ElementwiseBinary : (name : String) -> Graph -> Graph -> Graph
  ElementwiseUnary : (name : String) -> Graph -> Graph
  Select : Graph -> Graph -> Graph -> Graph
  Cond : Graph -> Graph -> Graph -> Graph -> Graph -> Graph
  Dot : Graph -> Graph -> Graph
  Cholesky : Graph -> Graph
  TriangularSolve : (lower : Bool) -> Graph -> Graph -> Graph
  RngBitGenerator : RandomAlgorithm -> Graph -> Shape -> Graph

Eq RandomAlgorithm where
  RngDefault == RngDefault = True
  RngThreeFry == RngThreeFry = True
  RngPhilox == RngPhilox = True
  _ == _ = False

export covering
Eq Graph where
  (FromLiteral {dtype} hash shape) == (FromLiteral {dtype=dtype'} hash' shape') =
     (typeString {dtype}, shape, hash) == (typeString {dtype=dtype'}, shape', hash')
  (Parameter {dtype} shape position) == (Parameter {dtype=dtype'} shape' position') =
    (typeString {dtype}, shape, position) == (typeString {dtype=dtype'}, shape', position')
  (MinFiniteValue {dtype}) == (MinFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxFiniteValue {dtype}) == (MaxFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (ConvertElementType {dtype} operand) == (ConvertElementType {dtype=dtype'} operand') =
    (typeString {dtype}, operand) == (typeString {dtype=dtype'}, operand')
  (GetTupleElement tuple index) == (GetTupleElement tuple' index') =
    (tuple, index) == (tuple', index')
  (Reshape to x) == (Reshape to' x') = (to, x) == (to', x')
  (Slice axis from to x) == (Slice axis' from' to' x') =
    (axis, from, to, x) == (axis', from', to', x')
  (Concat axis x y) == (Concat axis' x' y') = (axis, x, y) == (axis', x', y')
  (Diag x) == (Diag x') = x == x'
  (Triangle lower x) == (Triangle lower' x') = (lower, x) == (lower', x')
  (Transpose x) == (Transpose x') = x == x'
  (Identity {dtype} n) == (Identity {dtype=dtype'} n') =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  (Broadcast to x) == (Broadcast to' x') = (to, x) == (to', x')
  (Map f xs) == (Map f' xs') = (f, xs) == (f', xs')
  (Reduce monoid axis x) == (Reduce monoid' axis' x') =
    (monoid, axis, x) == (monoid', axis', x')
  (Sort operands comparator dimension isStable) == (Sort operands' comparator' dimension' isStable')
    = (operands, comparator, dimension, isStable) == (operands', comparator', dimension', isStable')
  (ElementwiseBinary name x y) == (ElementwiseBinary name' x' y') = (name, x, y) == (name', x', y')
  (ElementwiseUnary name x) == (ElementwiseUnary name' x') = (name, x) == (name', x')
  (Select pred f t) == (Select pred' f' t') = (pred, f, t) == (pred', f', t')
  (Cond pred fTrue true fFalse false) == (Cond pred' fTrue' true' fFalse' false') =
    (pred, fTrue, true, fFalse, false) == (pred', fTrue', true', fFalse', false')
  (Dot x y) == (Dot x' y') = (x, y) == (x', y')
  (Cholesky x) == (Cholesky x') = x == x'
  (TriangularSolve lower x y) == (TriangularSolve lower' x' y') = (lower, x, y) == (lower', x', y')
  (RngBitGenerator algorithm initialState shape)
    == (RngBitGenerator algorithm' initialState' shape') =
      (algorithm, initialState, shape) == (algorithm', initialState', shape')
  _ == _ = False

Hashable RandomAlgorithm where
  hashWithSalt salt algorithm = hashWithSalt salt $ the Int $ case algorithm of
    RngDefault => 0
    RngThreeFry => 1
    RngPhilox => 2

export
Hashable Graph where
  hashWithSalt salt (FromLiteral {dtype} hash shape) =
    salt `hashWithSalt` ("FromLiteral", typeString {dtype}, shape, hash)
  hashWithSalt salt (Parameter {dtype} shape position) =
    salt `hashWithSalt` ("Parameter", typeString {dtype}, shape, position)
  hashWithSalt salt (MinFiniteValue {dtype}) =
    salt `hashWithSalt` ("MinFiniteValue", typeString {dtype})
  hashWithSalt salt (MaxFiniteValue {dtype}) =
    salt `hashWithSalt` ("MaxFiniteValue", typeString {dtype})
  hashWithSalt salt (ConvertElementType {dtype} operand) =
    salt `hashWithSalt` ("ConvertElementType", typeString {dtype}) `hashWithSalt` operand
  hashWithSalt salt (GetTupleElement tuple index) =
    hashWithSalt salt ("GetTupleElement", index) `hashWithSalt` tuple
  hashWithSalt salt (Reshape to x) = salt `hashWithSalt` ("Reshape", to) `hashWithSalt` x
  hashWithSalt salt (Slice axis from to x) =
    salt `hashWithSalt` ("Slice", axis, from, to) `hashWithSalt` x
  hashWithSalt salt (Concat axis x y) =
    salt `hashWithSalt` ("Concat", axis) `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (Diag x) = salt `hashWithSalt` "Diag" `hashWithSalt` x
  hashWithSalt salt (Triangle lower x) = salt `hashWithSalt` ("Triangle", lower) `hashWithSalt` x
  hashWithSalt salt (Transpose x) = salt `hashWithSalt` "Transpose" `hashWithSalt` x
  hashWithSalt salt (Identity {dtype} n) = salt `hashWithSalt` ("Identity", typeString {dtype}, n)
  hashWithSalt salt (Broadcast to x) = salt `hashWithSalt` ("Broadcast", to) `hashWithSalt` x
  hashWithSalt salt (Map f xs) =
    let salt' = salt `hashWithSalt` "Map" `hashWithSalt` f
     in assert_total $ salt' `hashWithSalt` xs
  hashWithSalt salt (Reduce monoid axis x) =
    salt `hashWithSalt` "Reduce" `hashWithSalt` monoid `hashWithSalt` axis `hashWithSalt` x
  hashWithSalt salt (Sort operands comparator dimension isStable) =
    let salt' = salt `hashWithSalt` "Sort"
        salt'' = assert_total $ salt' `hashWithSalt` operands
     in salt'' `hashWithSalt` (dimension, isStable)
  hashWithSalt salt (Reverse axes operand) =
    salt `hashWithSalt` ("Reverse", axes) `hashWithSalt` operand
  hashWithSalt salt (ElementwiseBinary name x y) =
    salt `hashWithSalt` name `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (ElementwiseUnary name x) = salt `hashWithSalt` name `hashWithSalt` x
  hashWithSalt salt (Select pred f t) =
    salt `hashWithSalt` "Select" `hashWithSalt` pred `hashWithSalt` f `hashWithSalt` t
  hashWithSalt salt (Cond pred fTrue true fFalse false) = salt
    `hashWithSalt` "Cond"
    `hashWithSalt` pred
    `hashWithSalt` fTrue
    `hashWithSalt` true
    `hashWithSalt` fFalse
    `hashWithSalt` false
  hashWithSalt salt (Dot x y) = salt `hashWithSalt` "Dot" `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (Cholesky x) = salt `hashWithSalt` "Cholesky" `hashWithSalt` x
  hashWithSalt salt (TriangularSolve lower x y) =
    salt `hashWithSalt` ("TriangularSolve", lower) `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (RngBitGenerator algorithm initialState shape) = salt
    `hashWithSalt` ("RngBitGenerator", algorithm)
    `hashWithSalt` initialState
    `hashWithSalt` shape
