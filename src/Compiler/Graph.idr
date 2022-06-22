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

import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Types
import Util.Hashable

||| A `Graph` represents a computational graph used to compute a tensor value. It is defined by
||| the following property: For any two `Graph`s gx and gy that compute tensors x and y respectively,
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
  UniformFloatingPointDistributionValue :
    Graph -> Graph -> BitGenerator -> Graph -> Graph -> Shape -> Graph
  UniformFloatingPointDistributionState :
    Graph -> Graph -> BitGenerator -> Graph -> Graph -> Shape -> Graph
  NormalFloatingPointDistributionValue : Graph -> Graph -> BitGenerator -> Shape -> Graph
  NormalFloatingPointDistributionState : Graph -> Graph -> BitGenerator -> Shape -> Graph

Eq BitGenerator where
  ThreeFry == ThreeFry = True
  Philox == Philox = True
  _ == _ = False

export
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
    assert_total $ (typeString {dtype}, operand) == (typeString {dtype=dtype'}, operand')
  (GetTupleElement tuple index) == (GetTupleElement tuple' index') =
    assert_total $ (tuple, index) == (tuple', index')
  (Reshape to x) == (Reshape to' x') = to == to' && x == x'
  (Slice axis from to x) == (Slice axis' from' to' x') =
    assert_total $ (axis, from, to, x) == (axis', from', to', x')
  (Concat axis x y) == (Concat axis' x' y') = axis == axis' && x == x' && y == y'
  (Diag x) == (Diag x') = x == x'
  (Triangle lower x) == (Triangle lower' x') = lower == lower' && x == x'
  (Transpose x) == (Transpose x') = x == x'
  (Identity {dtype} n) == (Identity {dtype=dtype'} n') =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  (Broadcast to x) == (Broadcast to' x') = to == to' && x == x'
  (Map f xs) == (Map f' xs') = f == f' && (assert_total $ xs == xs')
  (Reduce monoid axis x) == (Reduce monoid' axis' x') =
    monoid == monoid' && axis == axis' && x == x'
  (Sort operands comparator dimension isStable)
    == (Sort operands' comparator' dimension' isStable') =
      (assert_total $ operands == operands')
      && comparator == comparator'
      && dimension == dimension'
      &&  isStable == isStable'
  (ElementwiseBinary name x y) == (ElementwiseBinary name' x' y') =
    name == name' && x == x' && y == y'
  (ElementwiseUnary name x) == (ElementwiseUnary name' x') = name == name' && x == x'
  (Select pred f t) == (Select pred' f' t') = pred == pred' && f == f' && t == t'
  (Cond pred fTrue true fFalse false) == (Cond pred' fTrue' true' fFalse' false') =
    pred == pred' && fTrue == fTrue' && true == true' && fFalse == fFalse' && false == false'
  (Dot x y) == (Dot x' y') = x == x' && y == y'
  (Cholesky x) == (Cholesky x') = x == x'
  (TriangularSolve lower x y) == (TriangularSolve lower' x' y') =
    lower == lower' && x == x' && y == y'
  (UniformFloatingPointDistributionValue key initialState bitGenerator minval maxval shape) ==
    (UniformFloatingPointDistributionValue key' initialState' bitGenerator' minval' maxval' shape')
      = key == key'
        && initialState == initialState'
        && bitGenerator == bitGenerator'
        && minval == minval'
        && maxval == maxval'
  (UniformFloatingPointDistributionState key initialState bitGenerator minval maxval shape) ==
    (UniformFloatingPointDistributionState key' initialState' bitGenerator' minval' maxval' shape')
      = key == key'
        && initialState == initialState'
        && bitGenerator == bitGenerator'
        && minval == minval'
        && maxval == maxval'
  (NormalFloatingPointDistributionValue key initialState bitGenerator shape) ==
    (NormalFloatingPointDistributionValue key' initialState' bitGenerator' shape')
      = key == key' && initialState == initialState' && bitGenerator == bitGenerator'
  (NormalFloatingPointDistributionState key initialState bitGenerator shape) ==
    (NormalFloatingPointDistributionState key' initialState' bitGenerator' shape')
      = key == key' && initialState == initialState' && bitGenerator == bitGenerator'
  _ == _ = False

Hashable BitGenerator where
  hashWithSalt salt algorithm = hashWithSalt salt $ the Int $ case algorithm of
    ThreeFry => 1
    Philox => 2

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
  hashWithSalt salt
    (UniformFloatingPointDistributionValue key initialState bitGenerator minval maxval shape) = salt
      `hashWithSalt` "UniformFloatingPointDistributionValue"
      `hashWithSalt` key
      `hashWithSalt` initialState
      `hashWithSalt` bitGenerator
      `hashWithSalt` minval
      `hashWithSalt` maxval
      `hashWithSalt` shape
  hashWithSalt salt
    (UniformFloatingPointDistributionState key initialState bitGenerator minval maxval shape) = salt
      `hashWithSalt` "UniformFloatingPointDistributionState"
      `hashWithSalt` key
      `hashWithSalt` initialState
      `hashWithSalt` bitGenerator
      `hashWithSalt` minval
      `hashWithSalt` maxval
      `hashWithSalt` shape
  hashWithSalt salt
    (NormalFloatingPointDistributionValue key initialState bitGenerator shape) = salt
      `hashWithSalt` "NormalFloatingPointDistributionValue"
      `hashWithSalt` key
      `hashWithSalt` initialState
      `hashWithSalt` bitGenerator
      `hashWithSalt` shape
  hashWithSalt salt
    (NormalFloatingPointDistributionState key initialState bitGenerator shape) = salt
      `hashWithSalt` "NormalFloatingPointDistributionState"
      `hashWithSalt` key
      `hashWithSalt` initialState
      `hashWithSalt` bitGenerator
      `hashWithSalt` shape
