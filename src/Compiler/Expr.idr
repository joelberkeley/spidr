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

import Data.SortedMap
import Decidable.Equality

import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

public export
data ShapeAndType : Type where
  MkShapeAndType : Shape -> (0 dtype : Type) -> Primitive dtype => ShapeAndType

data Expr : Type where

public export 0
Env : Type
Env = SortedMap Nat Expr

public export
data Fn : Nat -> Type where
  MkFn : {arity : _} -> Vect arity (Nat, ShapeAndType) -> Nat -> Env -> Fn arity

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
  Arg : Nat -> Expr
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
