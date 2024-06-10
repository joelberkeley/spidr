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
module Compiler.Expr

import Decidable.Equality

import Control.Monad.State
import Compiler.LiteralRW
import Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

public export
data ShapeAndType : Type where
  MkShapeAndType : Shape -> (0 dtype : Type) -> Primitive dtype => ShapeAndType

public export
data Expr : Type where

-- we use `List (Nat, Expr)` for O(1) append (all we do when building the graph is append)
-- we can't use `(Nat, List Expr)`, or even better `(n ** Vect n Expr)`, because we don't handle
-- scoping properly so node pointers aren't contiguous and don't match list indices
export
data Env = MkEnv Nat (List (Nat, Expr))

export
empty : Env
empty = MkEnv 0 []

export
toList : Env -> (Nat, List (Nat, Expr))
toList (MkEnv n env) = (n, reverse env)

public export
data Fn : Nat -> Type where

  ||| @arity The function arity.
  ||| @params The function parameter shapes and dtypes.
  ||| @result The function result.
  ||| @env Bindings within the function. Includes only nodes in this scope, not outer or inner scope.
  MkFn : {arity : _} ->
         (params : Vect arity ShapeAndType) ->
         (result : Expr) ->
         (env : Env) ->
         Fn arity

public export
data BinaryOp =
    Eq
  | Ne
  | Add
  | Sub
  | Mul
  | Div
  | Rem
  | Pow
  | Lt
  | Gt
  | Le
  | Ge
  | And
  | Or
  | Min
  | Max

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
  Var : Nat -> Expr
  Arg : Nat -> Expr
  Tuple : List Expr -> Expr
  GetTupleElement : Nat -> Expr -> Expr
  MinValue : Primitive dtype => Expr
  MaxValue : Primitive dtype => Expr
  MinFiniteValue : Primitive dtype => Expr
  MaxFiniteValue : Primitive dtype => Expr
  Iota : Primitive dtype => Shape -> Nat -> Expr
  ConvertElementType : Primitive dtype => Expr -> Expr
  Reshape : Shape -> Shape -> Expr -> Expr
  Slice : List Nat -> List Nat -> List Nat -> Expr -> Expr
  DynamicSlice : List Expr -> List Nat -> Expr -> Expr
  Concat : Nat -> Expr -> Expr -> Expr
  Diag : Expr -> Expr
  Triangle : (lower : Bool) -> Expr -> Expr
  Transpose : List Nat -> Expr -> Expr
  Identity : Primitive dtype => Nat -> Expr
  Broadcast : Primitive dtype => Shape -> Shape -> Expr -> Expr

  ||| Apply function `f` with given `arity` over `args`.
  |||
  ||| @f The function to apply.
  ||| @args The arguments to apply `f` to.
  Map : (f : Fn arity) -> (args : Vect arity Expr) -> Shape -> Expr

  Reduce : Fn 2 -> Expr -> List Nat -> Expr -> Expr
  Sort : Fn 2 -> Nat -> Bool -> List Expr -> Expr
  Reverse : List Nat -> Expr -> Expr
  BinaryElementwise : BinaryOp -> Expr -> Expr -> Expr
  UnaryElementwise : UnaryOp -> Expr -> Expr
  Argmin : Primitive out => Nat -> Expr -> Expr
  Argmax : Primitive out => Nat -> Expr -> Expr
  Select : Expr -> Expr -> Expr -> Expr
  Cond : Expr -> Fn 1 -> Expr -> Fn 1 -> Expr -> Expr
  Dot : Expr -> Expr -> Expr
  DotGeneral : (lBatch, lContract, rBatch, rContract : List Nat) -> Expr -> Expr -> Expr
  Cholesky : Expr -> Expr
  TriangularSolve : Expr -> Expr -> Bool -> Expr
  UniformFloatingPoint : Expr -> Expr -> Expr -> Expr -> Shape -> Expr
  NormalFloatingPoint : Expr -> Expr -> Shape -> Expr

export
addNode : Expr -> State Env Expr
addNode expr = do
  MkEnv next env <- get
  put $ MkEnv (S next) ((next, expr) :: env)
  pure (Var next)

public export 0
FnExpr : Nat -> Type
FnExpr 0 = State Env Expr
FnExpr (S k) = Expr -> FnExpr k

applyN : FnExpr arity -> Vect arity Nat -> State Env Expr
applyN f [] = f
applyN f (x :: xs) = applyN (f $ Var x) xs
