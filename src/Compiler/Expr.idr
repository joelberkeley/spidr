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

import Control.Monad.State
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

infix 9 ###

public export
data FullShape : Type where
  (###) : Shape -> (0 dtype : Type) -> Primitive dtype => FullShape

export
new : Ref Nat
new = do
  n <- get
  put (S n)
  pure n

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
         (params : Vect arity (Nat, FullShape)) ->
         (result : Nat) ->
         (env : Env) ->
         Fn arity

public export 0
ProgramShape : Type
ProgramShape = SortedMap Nat Shape

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

export
Show BinaryOp where
  show Eq = "Eq"
  show Ne = "Ne"
  show Add = "Add"
  show Sub = "Sub"
  show Mul = "Mul"
  show Div = "Div"
  show Rem = "Rem"
  show Pow = "Pow"
  show Lt = "Lt"
  show Gt = "Gt"
  show Le = "Le"
  show Ge = "Ge"
  show And = "And"
  show Or = "Or"
  show Min = "Min"
  show Max = "Max"

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

Show UnaryOp where
  show Not = "Not"
  show Neg = "Neg"
  show Reciprocal = "Reciprocal"
  show Ceil = "Ceil"
  show Floor = "Floor"
  show Abs = "Abs"
  show Log = "Log"
  show Exp = "Exp"
  show Logistic = "Logistic"
  show Erf = "Erf"
  show Square = "Square"
  show Sqrt = "Sqrt"
  show Sin = "Sin"
  show Cos = "Cos"
  show Tan = "Tan"
  show Asin = "Asin"
  show Acos = "Acos"
  show Atan = "Atan"
  show Sinh = "Sinh"
  show Cosh = "Cosh"
  show Tanh = "Tanh"
  show Asinh = "Asinh"
  show Acosh = "Acosh"
  show Atanh = "Atanh"

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
  Reduce : Fn 2 -> Nat -> List Nat -> Nat -> Expr
  Sort : Fn 2 -> Nat -> Bool -> List Nat -> Expr
  Reverse : List Nat -> Nat -> Expr
  UnaryElementwise : {shape : Shape} -> UnaryOp -> Nat -> Expr
  BinaryElementwise : {shape : Shape} -> BinaryOp -> Nat -> Nat -> Expr
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

public export 0
FnExpr : Nat -> Type
FnExpr 0 = State Env Nat
FnExpr (S k) = Nat -> FnExpr k

applyN : FnExpr arity -> Vect arity Nat -> State Env Nat
applyN f [] = f
applyN f (x :: xs) = applyN (f x) xs

export
addFn : {arity : _} -> Vect arity FullShape -> FnExpr arity -> State Env (Fn arity)
addFn params f = do
  MkEnv next env <- get
  let (subEnv@(MkEnv next _), params, result) = runState (MkEnv next []) $ do
        xs <- traverse addArg params
        result <- applyN f xs
        pure (zip xs params, result)
  put (MkEnv next env)
  pure (MkFn params result subEnv)

  where
  addArg : FullShape -> State Env Nat
  addArg st = do
    MkEnv next env <- get
    put (MkEnv (S next) ((next, Arg next) :: env))
    pure next

export
Show Expr where
  show (FromLiteral {shape} _) = "FromLiteral {shape = \{show shape}}"
  show (Arg i) = "Arg \{show i}"
  show (Diag i) = "Diag \{show i}"
  show (Reshape from to i) = "Reshape \{show from} \{show to} \{show i}"
  show (Broadcast from to i) = "Broadcast \{show from} \{show to} \{show i}"
  show (UnaryElementwise {shape} op i) = "UnaryElementwise \{show op} \{show i}"
  show (BinaryElementwise {shape} op i j) = "BinaryElementwise \{show op} \{show i} \{show j}"
  show (Concat axis x y) = "Concat \{show axis} \{show x} \{show y}"
  show _ = "OtherExpr"
