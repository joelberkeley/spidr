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
module Compiler.Expr

import Data.Hashable

-- Expr can be independent of XLA
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData

import Literal
import Primitive
import Types
import Util
import Util.Hashable

public export
data Expr : Type where
  FromLiteral : PrimitiveRW dtype ty => Primitive dtype => {shape : _} -> Literal shape ty -> Expr
  Parameter : Primitive dtype => Nat -> Types.Shape -> String -> Expr
  MinFiniteValue : Primitive dtype => Expr
  MaxFiniteValue : Primitive dtype => Expr
  ConvertElementType : Primitive dtype => Expr -> Expr
  Reshape : Types.Shape -> Types.Shape -> Expr -> Expr
  Slice : List Nat -> List Nat -> List Nat -> Expr -> Expr
  Concat : Nat -> Expr -> Expr -> Expr
  Diag : Expr -> Expr
  Triangle : (lower : Bool) -> Expr -> Expr
  Transpose : Expr -> Expr
  Identity : Primitive dtype => Nat -> Expr
  Broadcast : Primitive dtype => Types.Shape -> Types.Shape -> Expr -> Expr
  Map : List Expr -> Expr -> List Expr -> Types.Shape -> Expr
  Reduce : Expr -> Expr -> Expr -> Expr -> Nat -> Expr -> Expr
  Sort : Expr -> Expr -> Expr -> Nat -> Bool -> List Expr -> Expr
  Reverse : List Nat -> Expr -> Expr
  Eq : Expr -> Expr -> Expr
  Ne : Expr -> Expr -> Expr
  Add : Expr -> Expr -> Expr
  Sub : Expr -> Expr -> Expr
  Mul : Expr -> Expr -> Expr
  Div : Expr -> Expr -> Expr
  Pow : Expr -> Expr -> Expr
  Lt : Expr -> Expr -> Expr
  Gt : Expr -> Expr -> Expr
  Le : Expr -> Expr -> Expr
  Ge : Expr -> Expr -> Expr
  And : Expr -> Expr -> Expr
  Or : Expr -> Expr -> Expr
  Min : Expr -> Expr -> Expr
  Max : Expr -> Expr -> Expr
  Not : Expr -> Expr
  Neg : Expr -> Expr
  Reciprocal : Expr -> Expr
  Abs : Expr -> Expr
  Ceil : Expr -> Expr
  Floor : Expr -> Expr
  Exp : Expr -> Expr
  Erf : Expr -> Expr
  Log : Expr -> Expr
  Logistic : Expr -> Expr
  Square : Expr -> Expr
  Sqrt : Expr -> Expr
  Sin : Expr -> Expr
  Cos : Expr -> Expr
  Tan : Expr -> Expr
  Asin : Expr -> Expr
  Acos : Expr -> Expr
  Atan : Expr -> Expr
  Sinh : Expr -> Expr
  Cosh : Expr -> Expr
  Tanh : Expr -> Expr
  Asinh : Expr -> Expr
  Acosh : Expr -> Expr
  Atanh : Expr -> Expr
  Select : Expr -> Expr -> Expr -> Expr
  Cond : Expr -> Expr -> Expr -> Expr -> Expr -> Expr -> Expr -> Expr
  Dot : Expr -> Expr -> Expr
  Cholesky : Expr -> Expr
  TriangularSolve : Expr -> Expr -> Bool -> Bool -> Bool -> Transpose -> Expr
  UniformFloatingPointDistributionValue :
    Expr -> Expr -> BitGenerator -> Expr -> Expr -> Types.Shape -> Expr
  UniformFloatingPointDistributionState :
    Expr -> Expr -> BitGenerator -> Expr -> Expr -> Types.Shape -> Expr
  NormalFloatingPointDistributionValue : Expr -> Expr -> BitGenerator -> Types.Shape -> Expr
  NormalFloatingPointDistributionState : Expr -> Expr -> BitGenerator -> Types.Shape -> Expr

Prelude.Eq BitGenerator where
  ThreeFry == ThreeFry = True
  Philox == Philox = True
  _ == _ = False

Prelude.Eq Transpose where
  NoTranspose == NoTranspose = True
  Transpose_ == Transpose_ = True
  Adjoint == Adjoint = True
  _ == _ = False

export
Prelude.Eq Expr where
  (FromLiteral {dtype} lit {shape}) == (FromLiteral {dtype=dtype'} lit' {shape=shape'}) =
    (typeString {dtype}, shape, hash lit) == (typeString {dtype=dtype'}, shape', hash lit')
  (Parameter {dtype} position shape name) == (Parameter {dtype=dtype'} position' shape' name') =
    (typeString {dtype}, position, shape, name) ==
      (typeString {dtype=dtype'}, position', shape', name')
  (MinFiniteValue {dtype}) == (MinFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxFiniteValue {dtype}) == (MaxFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (ConvertElementType {dtype} operand) == (ConvertElementType {dtype=dtype'} operand') =
    assert_total $ (typeString {dtype}, operand) == (typeString {dtype=dtype'}, operand')
  (Reshape from to x) == (Reshape from' to' x') = (from, to) == (from', to') && x == x'
  (Slice starts stops strides x) == (Slice starts' stops' strides' x') =
    (starts, stops, strides) == (starts', stops', strides') && x == x'
  (Concat axis x y) == (Concat axis' x' y') = axis == axis' && x == x' && y == y'
  (Diag x) == (Diag x') = x == x'
  (Triangle lower x) == (Triangle lower' x') = lower == lower' && x == x'
  (Transpose x) == (Transpose x') = x == x'
  (Identity {dtype} n) == (Identity {dtype=dtype'} n') =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  (Broadcast from to x) == (Broadcast from' to' x') = (from, to) == (from', to') && x == x'
  (Map params f xs dims) == (Map params' f' xs' dims') =
    (assert_total $ params == params') && f == f' && (assert_total $ xs == xs') && dims == dims'
  (Reduce p0 p1 monoid neutral axis x) == (Reduce p0' p1' monoid' neutral' axis' x') =
    p0 == p0' && p1 == p1' && monoid == monoid' && neutral == neutral' && axis == axis' && x == x'
  (Sort p0 p1 comparator dimension isStable operands)
    == (Sort p0' p1' comparator' dimension' isStable' operands') =
      p0 == p0'
      && p1 == p1'
      && comparator == comparator'
      && dimension == dimension'
      && isStable == isStable'
      && (assert_total $ operands == operands')
  (Eq l r) == (Eq l' r') = l == l' && r == r'
  (Ne l r) == (Ne l' r') = l == l' && r == r'
  (Add l r) == (Add l' r') = l == l' && r == r'
  (Sub l r) == (Sub l' r') = l == l' && r == r'
  (Mul l r) == (Mul l' r') = l == l' && r == r'
  (Div l r) == (Div l' r') = l == l' && r == r'
  (Pow l r) == (Pow l' r') = l == l' && r == r'
  (Lt l r) == (Lt l' r') = l == l' && r == r'
  (Gt l r) == (Gt l' r') = l == l' && r == r'
  (Le l r) == (Le l' r') = l == l' && r == r'
  (Ge l r) == (Ge l' r') = l == l' && r == r'
  (And l r) == (And l' r') = l == l' && r == r'
  (Or l r) == (Or l' r') = l == l' && r == r'
  (Min l r) == (Min l' r') = l == l' && r == r'
  (Max l r) == (Max l' r') = l == l' && r == r'
  (Not expr) == (Not expr') = expr == expr'
  (Neg expr) == (Neg expr') = expr == expr'
  (Reciprocal expr) == (Reciprocal expr') = expr == expr'
  (Abs expr) == (Abs expr') = expr == expr'
  (Ceil expr) == (Ceil expr') = expr == expr'
  (Floor expr) == (Floor expr') = expr == expr'
  (Log expr) == (Log expr') = expr == expr'
  (Exp expr) == (Exp expr') = expr == expr'
  (Logistic expr) == (Logistic expr') = expr == expr'
  (Erf expr) == (Erf expr') = expr == expr'
  (Square expr) == (Square expr') = expr == expr'
  (Sqrt expr) == (Sqrt expr') = expr == expr'
  (Sin expr) == (Sin expr') = expr == expr'
  (Cos expr) == (Cos expr') = expr == expr'
  (Tan expr) == (Tan expr') = expr == expr'
  (Asin expr) == (Asin expr') = expr == expr'
  (Acos expr) == (Acos expr') = expr == expr'
  (Atan expr) == (Atan expr') = expr == expr'
  (Sinh expr) == (Sinh expr') = expr == expr'
  (Cosh expr) == (Cosh expr') = expr == expr'
  (Tanh expr) == (Tanh expr') = expr == expr'
  (Asinh expr) == (Asinh expr') = expr == expr'
  (Acosh expr) == (Acosh expr') = expr == expr'
  (Atanh expr) == (Atanh expr') = expr == expr'
  (Select pred f t) == (Select pred' f' t') = pred == pred' && f == f' && t == t'
  (Cond pred pt fTrue true pf fFalse false) == (Cond pred' pt' fTrue' true' pf' fFalse' false') =
    pred == pred'
    && pt == pt'
    && fTrue == fTrue'
    && true == true'
    && pf == pf'
    && fFalse == fFalse'
    && false == false'
  (Dot x y) == (Dot x' y') = x == x' && y == y'
  (Cholesky x) == (Cholesky x') = x == x'
  (TriangularSolve x y leftSide lower unitDiagonal transposeA) ==
    (TriangularSolve x' y' leftSide' lower' unitDiagonal' transposeA') =
      x == x' && y == y' &&
        (leftSide, lower, unitDiagonal, transposeA) == (leftSide', lower', unitDiagonal', transposeA')
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
  hashWithSalt salt bitGenerator = hashWithSalt salt (cast {to=Int} bitGenerator)

Hashable Transpose where
  hashWithSalt salt NoTranspose = hashWithSalt salt 0
  hashWithSalt salt Transpose_ = hashWithSalt salt 1
  hashWithSalt salt Adjoint = hashWithSalt salt 2

export
Hashable Expr where
  hashWithSalt salt (FromLiteral {shape} {dtype} lit) =
    salt `hashWithSalt` ("FromLiteral", typeString {dtype}, shape, lit)
  hashWithSalt salt (Parameter {dtype} position shape name) =
    salt `hashWithSalt` ("Parameter", typeString {dtype}, shape, position, name)
  hashWithSalt salt (MinFiniteValue {dtype}) =
    salt `hashWithSalt` ("MinFiniteValue", typeString {dtype})
  hashWithSalt salt (MaxFiniteValue {dtype}) =
    salt `hashWithSalt` ("MaxFiniteValue", typeString {dtype})
  hashWithSalt salt (ConvertElementType {dtype} operand) =
    salt `hashWithSalt` ("ConvertElementType", typeString {dtype}) `hashWithSalt` operand
  hashWithSalt salt (Reshape from to x) =
    salt `hashWithSalt` ("Reshape", from, to) `hashWithSalt` x
  hashWithSalt salt (Slice starts stops strides x) =
    salt `hashWithSalt` ("Slice", starts, stops, strides) `hashWithSalt` x
  hashWithSalt salt (Concat axis x y) =
    salt `hashWithSalt` ("Concat", axis) `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (Diag x) = salt `hashWithSalt` "Diag" `hashWithSalt` x
  hashWithSalt salt (Triangle lower x) = salt `hashWithSalt` ("Triangle", lower) `hashWithSalt` x
  hashWithSalt salt (Transpose x) = salt `hashWithSalt` "Transpose" `hashWithSalt` x
  hashWithSalt salt (Identity {dtype} n) = salt `hashWithSalt` ("Identity", typeString {dtype}, n)
  hashWithSalt salt (Broadcast from to x) =
    salt `hashWithSalt` ("Broadcast", from, to) `hashWithSalt` x
  hashWithSalt salt (Map params f xs dims) =
    let salt = salt `hashWithSalt` "Map"
        salt = assert_total $ salt `hashWithSalt` params
        salt = salt `hashWithSalt` f
        salt = assert_total $ salt `hashWithSalt` xs
     in salt `hashWithSalt` dims
  hashWithSalt salt (Reduce p0 p1 monoid neutral axis x) = salt
    `hashWithSalt` "Reduce"
    `hashWithSalt` p0
    `hashWithSalt` p1
    `hashWithSalt` monoid
    `hashWithSalt` neutral 
    `hashWithSalt` axis
    `hashWithSalt` x
  hashWithSalt salt (Sort p0 p1 comparator dimension isStable operands) =
    let salt = salt
          `hashWithSalt` "Sort"
          `hashWithSalt` p0
          `hashWithSalt` p1
          `hashWithSalt` (dimension, isStable)
     in assert_total $ salt `hashWithSalt` operands
  hashWithSalt salt (Reverse axes operand) =
    salt `hashWithSalt` ("Reverse", axes) `hashWithSalt` operand
  hashWithSalt salt (Eq l r) = salt `hashWithSalt` "Eq" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Ne l r) = salt `hashWithSalt` "Ne" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Add l r) = salt `hashWithSalt` "Add" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Sub l r) = salt `hashWithSalt` "Sub" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Mul l r) = salt `hashWithSalt` "Mul" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Div l r) = salt `hashWithSalt` "Div" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Pow l r) = salt `hashWithSalt` "Pow" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Lt l r) = salt `hashWithSalt` "Lt" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Gt l r) = salt `hashWithSalt` "Gt" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Le l r) = salt `hashWithSalt` "Le" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Ge l r) = salt `hashWithSalt` "Ge" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (And l r) = salt `hashWithSalt` "And" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Or l r) = salt `hashWithSalt` "Or" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Min l r) = salt `hashWithSalt` "Min" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Max l r) = salt `hashWithSalt` "Max" `hashWithSalt` l `hashWithSalt` r
  hashWithSalt salt (Not expr) = salt `hashWithSalt` "Not" `hashWithSalt` expr
  hashWithSalt salt (Neg expr) = salt `hashWithSalt` "Neg" `hashWithSalt` expr
  hashWithSalt salt (Reciprocal expr) = salt `hashWithSalt` "Reciprocal" `hashWithSalt` expr
  hashWithSalt salt (Abs expr) = salt `hashWithSalt` "Abs" `hashWithSalt` expr
  hashWithSalt salt (Ceil expr) = salt `hashWithSalt` "Ceil" `hashWithSalt` expr
  hashWithSalt salt (Floor expr) = salt `hashWithSalt` "Floor" `hashWithSalt` expr
  hashWithSalt salt (Log expr) = salt `hashWithSalt` "Log" `hashWithSalt` expr
  hashWithSalt salt (Exp expr) = salt `hashWithSalt` "Exp" `hashWithSalt` expr
  hashWithSalt salt (Logistic expr) = salt `hashWithSalt` "Logistic" `hashWithSalt` expr
  hashWithSalt salt (Erf expr) = salt `hashWithSalt` "Erf" `hashWithSalt` expr
  hashWithSalt salt (Square expr) = salt `hashWithSalt` "Square" `hashWithSalt` expr
  hashWithSalt salt (Sqrt expr) = salt `hashWithSalt` "Sqrt" `hashWithSalt` expr
  hashWithSalt salt (Sin expr) = salt `hashWithSalt` "Sin" `hashWithSalt` expr
  hashWithSalt salt (Cos expr) = salt `hashWithSalt` "Cos" `hashWithSalt` expr
  hashWithSalt salt (Tan expr) = salt `hashWithSalt` "Tan" `hashWithSalt` expr
  hashWithSalt salt (Asin expr) = salt `hashWithSalt` "Asin" `hashWithSalt` expr
  hashWithSalt salt (Acos expr) = salt `hashWithSalt` "Acos" `hashWithSalt` expr
  hashWithSalt salt (Atan expr) = salt `hashWithSalt` "Atan" `hashWithSalt` expr
  hashWithSalt salt (Sinh expr) = salt `hashWithSalt` "Sinh" `hashWithSalt` expr
  hashWithSalt salt (Cosh expr) = salt `hashWithSalt` "Cosh" `hashWithSalt` expr
  hashWithSalt salt (Tanh expr) = salt `hashWithSalt` "Tanh" `hashWithSalt` expr
  hashWithSalt salt (Asinh expr) = salt `hashWithSalt` "Asinh" `hashWithSalt` expr
  hashWithSalt salt (Acosh expr) = salt `hashWithSalt` "Acosh" `hashWithSalt` expr
  hashWithSalt salt (Atanh expr) = salt `hashWithSalt` "Atanh" `hashWithSalt` expr
  hashWithSalt salt (Select pred f t) =
    salt `hashWithSalt` "Select" `hashWithSalt` pred `hashWithSalt` f `hashWithSalt` t
  hashWithSalt salt (Cond pred pt fTrue true pf fFalse false) = salt
    `hashWithSalt` "Cond"
    `hashWithSalt` pred
    `hashWithSalt` pt
    `hashWithSalt` fTrue
    `hashWithSalt` true
    `hashWithSalt` pf
    `hashWithSalt` fFalse
    `hashWithSalt` false
  hashWithSalt salt (Dot x y) = salt `hashWithSalt` "Dot" `hashWithSalt` x `hashWithSalt` y
  hashWithSalt salt (Cholesky x) = salt `hashWithSalt` "Cholesky" `hashWithSalt` x
  hashWithSalt salt (TriangularSolve x y leftSide lower unitDiagonal transposeA) = salt
    `hashWithSalt` "TriangularSolve"
    `hashWithSalt` x
    `hashWithSalt` y
    `hashWithSalt` (leftSide, lower, unitDiagonal, transposeA)
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
