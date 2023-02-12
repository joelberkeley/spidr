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

import Decidable.Equality

import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

data ShapeAndType : Type where
  MkShapeAndType : Shape -> (0 dtype : Type) -> Primitive dtype => ShapeAndType

data Expr : Type where

public export
data Fn : Nat -> Type where
  MkFn : {arity : _} -> Vect arity ShapeAndType -> Expr -> Fn arity

public export
data BinaryOp = Eq | Ne | Add | Sub | Mul | Div | Pow | Lt | Gt | Le | Ge | And | Or | Min | Max

public export
data UnaryOp =
    Not
  | Neg
  | Reciprocal
  | Ceil
  | Floor
  | Abs
  | Log
  | Exp
  | Logistic
  | Erf
  | Square
  | Sqrt
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Asinh
  | Acosh
  | Atanh

public export
data Expr : Type where
  FromLiteral : PrimitiveRW dtype ty => {shape : _} -> Literal shape ty -> Expr
  Parameter : Primitive dtype => Nat -> Shape -> String -> Expr
  Tuple : List Nat -> Expr
  GetTupleElement : Nat -> Nat -> Expr
  MinValue : Primitive dtype => Expr
  MaxValue : Primitive dtype => Expr
  MinFiniteValue : Primitive dtype => Expr
  MaxFiniteValue : Primitive dtype => Expr
  ConvertElementType : Primitive dtype => Nat -> Expr
  Reshape : Shape -> Shape -> Nat -> Expr
  Slice : List Nat -> List Nat -> List Nat -> Nat -> Expr
  DynamicSlice : List Nat -> List Nat -> Nat -> Expr
  Concat : Nat -> Nat -> Nat -> Expr
  Diag : Nat -> Expr
  Triangle : (lower : Bool) -> Nat -> Expr
  Transpose : List Nat -> Nat -> Expr
  Identity : Primitive dtype => Nat -> Expr
  Broadcast : Primitive dtype => Shape -> Shape -> Nat -> Expr
  Map : Fn n -> Vect n Nat -> Shape -> Expr
  Reduce : Fn 2 -> Nat -> List Nat -> Nat -> Expr
  Sort : Fn 2 -> Nat -> Bool -> List Nat -> Expr
  Reverse : List Nat -> Nat -> Expr
  BinaryElementwise : BinaryOp -> Nat -> Nat -> Expr
  UnaryElementwise : UnaryOp -> Nat -> Expr
  Argmin : Primitive out => Nat -> Nat -> Expr
  Argmax : Primitive out => Nat -> Nat -> Expr
  Select : Nat -> Nat -> Nat -> Expr
  Cond : Nat -> Fn 1 -> Nat -> Fn 1 -> Nat -> Expr
  Dot : Nat -> Nat -> Expr
  Cholesky : Nat -> Expr
  TriangularSolve : Nat -> Nat -> Bool -> Expr
  UniformFloatingPoint : Nat -> Nat -> Nat -> Nat -> Shape -> Expr
  NormalFloatingPoint : Nat -> Nat -> Shape -> Expr

{-
export
Prelude.Eq Expr where
  (FromLiteral {dtype} lit {shape}) == (FromLiteral {dtype=dtype'} lit' {shape=shape'}) =
    (typeString {dtype}, shape, hash lit) == (typeString {dtype=dtype'}, shape', hash lit')
  (Parameter {dtype} position shape name) == (Parameter {dtype=dtype'} position' shape' name') =
    (typeString {dtype}, position, shape, name) ==
      (typeString {dtype=dtype'}, position', shape', name')
  (Tuple xs) == (Tuple xs') = assert_total $ xs == xs'
  (GetTupleElement idx tuple) == (GetTupleElement idx' tuple') = idx == idx' && tuple == tuple'
  (MinValue {dtype}) == (MinValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxValue {dtype}) == (MaxValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MinFiniteValue {dtype}) == (MinFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxFiniteValue {dtype}) == (MaxFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (ConvertElementType {dtype} operand) == (ConvertElementType {dtype=dtype'} operand') =
    typeString {dtype} == typeString {dtype=dtype'} && operand == operand'
  (Reshape from to x) == (Reshape from' to' x') = (from, to) == (from', to') && x == x'
  (Slice starts stops strides x) == (Slice starts' stops' strides' x') =
    (starts, stops, strides) == (starts', stops', strides') && x == x'
  (DynamicSlice starts sizes x) == (DynamicSlice starts' sizes' x') =
    (assert_total $ starts == starts') && sizes == sizes' && x == x'
  (Concat axis x y) == (Concat axis' x' y') = axis == axis' && x == x' && y == y'
  (Diag x) == (Diag x') = x == x'
  (Triangle lower x) == (Triangle lower' x') = lower == lower' && x == x'
  (Transpose ordering x) == (Transpose ordering' x') = ordering == ordering' && x == x'
  (Identity {dtype} n) == (Identity {dtype=dtype'} n') =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  (Broadcast from to x) == (Broadcast from' to' x') = (from, to) == (from', to') && x == x'
  (Map {n} (MkFn params f) xs dims) == (Map {n=n'} (MkFn params' f') xs' dims') =
    case decEq n n' of
      Yes eq =>
        (assert_total $ params == rewrite eq in params')
        && f == f'
        && (assert_total $ xs == rewrite eq in xs')
        && dims == dims'
      No _ => False
  (Reduce (MkFn [p0, p1] monoid) neutral axes x) ==
    (Reduce (MkFn [p0', p1'] monoid') neutral' axes' x') =
      p0 == p0' && p1 == p1' && monoid == monoid' && neutral == neutral' && axes == axes' && x == x'
  (Sort (MkFn [p0, p1] comparator) dimension isStable operands) ==
    (Sort (MkFn [p0', p1'] comparator') dimension' isStable' operands') =
      p0 == p0'
      && p1 == p1'
      && comparator == comparator'
      && dimension == dimension'
      && isStable == isStable'
      && (assert_total $ operands == operands')
  (Reverse axes expr) == (Reverse axes' expr') = axes == axes' && expr == expr'
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
  (Argmin {out} axis expr) == (Argmin {out=out'} axis' expr') =
    (typeString {dtype=out}, axis) == (typeString {dtype=out'}, axis) && expr == expr'
  (Argmax {out} axis expr) == (Argmax {out=out'} axis' expr') =
    (typeString {dtype=out}, axis) == (typeString {dtype=out'}, axis) && expr == expr'
  (Select pred f t) == (Select pred' f' t') = pred == pred' && f == f' && t == t'
  (Cond pred (MkFn [pt] fTrue) true (MkFn [pf] fFalse) false) ==
    (Cond pred' (MkFn [pt'] fTrue') true' (MkFn [pf'] fFalse') false') =
      pred == pred'
      && pt == pt'
      && fTrue == fTrue'
      && true == true'
      && pf == pf'
      && fFalse == fFalse'
      && false == false'
  (Dot x y) == (Dot x' y') = x == x' && y == y'
  (Cholesky x) == (Cholesky x') = x == x'
  (TriangularSolve x y lower) == (TriangularSolve x' y' lower') =
    x == x' && y == y' && lower == lower'
  (UniformFloatingPoint key initialState minval maxval shape) ==
    (UniformFloatingPoint key' initialState' minval' maxval' shape') =
      key == key' && initialState == initialState' && minval == minval' && maxval == maxval'
  (NormalFloatingPoint key initialState shape) == (NormalFloatingPoint key' initialState' shape') =
      key == key' && initialState == initialState'
  _ == _ = False
-}
