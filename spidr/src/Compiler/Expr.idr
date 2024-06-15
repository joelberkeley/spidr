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

import Derive.Prelude
import Language.Reflection

import Compiler.LiteralRW
import Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

%language ElabReflection

public export
data ShapeAndType : Type where
  MkShapeAndType : Shape -> (0 dtype : Type) -> Primitive dtype => ShapeAndType

%runElab derive "ShapeAndType" [Show]

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
addNode : Expr -> State Env Nat
addNode expr = do
  MkEnv next env <- get
  put $ MkEnv (S next) ((next, expr) :: env)
  pure next

export
toList : Env -> (Nat, List (Nat, Expr))
toList (MkEnv n env) = (n, reverse env)

public export
data Fn : Nat -> Type where

  ||| @arity The function arity.
  ||| @params The function parameter position in the graph, along with its shape and dtype.
  ||| @result The position of the function result in the graph.
  ||| @env The function graph. Includes only nodes in this scope, not outer or inner scope.
  MkFn : {arity : _} ->
         (params : Vect arity (Nat, ShapeAndType)) ->
         (result : Nat) ->
         (env : Env) ->
         Fn arity

public export
data BinaryOp =
  Eq | Ne | Lt | Gt | Le | Ge | And | Or | Add | Sub | Mul | Div | Rem | Pow | Min | Max

%runElab derive "BinaryOp" [Show]

public export
data UnaryOp =
    Not
  | Neg | Reciprocal | Ceil | Floor | Abs | Log | Exp | Logistic | Erf | Square | Sqrt
  | Sin | Cos | Tan | Asin | Acos | Atan | Sinh | Cosh | Tanh | Asinh | Acosh | Atanh

%runElab derive "UnaryOp" [Show]

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
  Iota : Primitive dtype => Shape -> Nat -> Expr
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

  ||| Apply function `f` with given `arity` over `args`.
  |||
  ||| @f The function to apply.
  ||| @args The arguments to apply `f` to.
  Map : (f : Fn arity) -> (args : Vect arity Nat) -> Shape -> Expr

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
  DotGeneral : (lBatch, lContract, rBatch, rContract : List Nat) -> Nat -> Nat -> Expr
  Cholesky : Nat -> Expr
  TriangularSolve : Nat -> Nat -> Bool -> Expr
  UniformFloatingPoint : Nat -> Nat -> Nat -> Nat -> Shape -> Expr
  NormalFloatingPoint : Nat -> Nat -> Shape -> Expr

Show Expr

export
Show Env where
  show e = assert_total $ show $ snd $ toList e

Show (Fn arity) where
  show (MkFn params result env) = "MkFn \{show params} \{show result} \{show env}"

export
Show Expr where
  show (FromLiteral {shape, dtype} x) = "FromLiteral \{show shape} \{show $ xlaIdentifier {dtype}}"
  show (Arg k) = "Arg \{show k}"
  show (Tuple ks) = "Tuple \{show ks}"
  show (GetTupleElement k j) = "GetTupleElement \{show k} \{show j}"
  show (MinValue {dtype}) = "MinValue {dtype = \{show $ xlaIdentifier {dtype}}}"
  show (MaxValue {dtype}) = "MaxValue {dtype = \{show $ xlaIdentifier {dtype}}}"
  show (MinFiniteValue {dtype}) = "MinFiniteValue {dtype = \{show $ xlaIdentifier {dtype}}}"
  show (MaxFiniteValue {dtype}) = "MaxFiniteValue {dtype = \{show $ xlaIdentifier {dtype}}}"
  show (Iota ks k) = "Iota \{show ks} \{show k}"
  show (ConvertElementType k) = "ConvertElementType \{show k}"
  show (Reshape ks js k) = "Reshape \{show ks} \{show js} \{show k}"
  show (Slice ks js is k) = "Slice \{show ks} \{show js} \{show is} \{show k}"
  show (DynamicSlice ks js k) = "Slice \{show ks} \{show js} \{show k}"
  show (Concat k j i) = "Concat \{show k} \{show j} \{show i}"
  show (Diag k) = "Diag \{show k}"
  show (Triangle lower k) = "Triangle {lower = \{show lower}} \{show k}"
  show (Transpose ks k) = "Transpose \{show ks} \{show k}"
  show (Identity k) = "Identity \{show k}"
  show (Broadcast ks js k) = "Broadcast \{show ks} \{show js} \{show k}"
  show (Map f args ks) = "Map \{show f} \{show args} \{show ks}"
  show (Reduce x k ks j) = "Reduce \{show x} \{show k} \{show ks} \{show j}"
  show (Sort x k y ks) = "Sort \{show x} \{show k} \{show y} \{show ks}"
  show (Reverse ks k) = "Reverse \{show ks} \{show k}"
  show (BinaryElementwise x k j) = "\{show x} \{show k} \{show j}"
  show (UnaryElementwise x k) = "\{show x} \{show k}"
  show (Argmin k j) = "Argmin \{show k} \{show j}"
  show (Argmax k j) = "Argmax \{show k} \{show j}"
  show (Select k j i) = "Select \{show k} \{show j} \{show i}"
  show (Cond k x j y i) = "Cond \{show k} \{show x} \{show j} \{show y} \{show i}"
  show (Dot k j) = "Dot \{show k} \{show j}"
  show (DotGeneral lBatch lContract rBatch rContract k j) =
    "DotGeneral \{show lBatch} \{show lContract} \{show rBatch} \{show rContract} \{show k} \{show j}"
  show (Cholesky k) = "Cholesky \{show k}"
  show (TriangularSolve k j x) = "TriangularSolve \{show k} \{show j} \{show x}"
  show (UniformFloatingPoint k j i k1 ks) =
    "UniformFloatingPoint \{show k} \{show j} \{show i} \{show k1} \{show ks}"
  show (NormalFloatingPoint k j ks) = "NormalFloatingPoint \{show k} \{show j} \{show ks}"

public export 0
FnExpr : Nat -> Type
FnExpr 0 = State Env Nat
FnExpr (S k) = Nat -> FnExpr k

applyN : FnExpr arity -> Vect arity Nat -> State Env Nat
applyN f [] = f
applyN f (x :: xs) = applyN (f x) xs

export
addFn : {arity : _} -> Vect arity ShapeAndType -> FnExpr arity -> State Env (Fn arity)
addFn params f = do
  MkEnv next env <- get
  let (subEnv@(MkEnv next _), params, result) = runState (MkEnv next []) $ do
        xs <- traverse addArg params
        result <- applyN f xs
        pure (zip xs params, result)
  put (MkEnv next env)
  pure (MkFn params result subEnv)

  where
  addArg : ShapeAndType -> State Env Nat
  addArg st = do
    MkEnv next env <- get
    put (MkEnv (S next) ((next, Arg next) :: env))
    pure next
