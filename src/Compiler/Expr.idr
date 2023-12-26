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
import Data.SortedMap
import Control.Monad.State
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

public export
data Expr : Type where

public export
data Fn : Nat -> Type

-- we use `List (Nat, a)` for O(1) append (all we do when building the graph is append)
-- we can't use `(Nat, List a)`, or even better `(n ** Vect n a)`, because we don't handle
-- scoping properly so node pointers aren't contiguous and don't match list indices
public export 0
TopSort : Type -> Type
TopSort a = (Nat, List (Nat, a))

-- perhaps a better option is to use a single list for both functions and nodes, by
-- combining them with a `data Node = F (a ** Fn a) | E Expr` ... this is so that we
-- can efficiently build nodes and functions in order
export
data Env = MkEnv (TopSort (arity ** Fn arity)) (TopSort Expr)

export
empty : Env
empty = MkEnv (0, []) (0, [])

export
addNode : Expr -> State Env Nat
addNode expr = do
  MkEnv children (next, env) <- get
  put $ MkEnv children (S next, (next, expr) :: env)
  pure next

export
toList : Env -> List (Nat, Expr)
toList (MkEnv _ (_, env)) = reverse env

export
findChild : Env -> Nat -> Maybe (a ** Fn a)
-- list indices don't correspond to nodes do they? Aren't we meant to
findChild (MkEnv (_, children) _) n = lookup n $ SortedMap.fromList children

export
childKeys : Env -> List Nat
childKeys (MkEnv (_, children) _) = keys $ SortedMap.fromList children

public export
data ShapeAndType : Type where
  MkShapeAndType : Shape -> (0 dtype : Type) -> Primitive dtype => ShapeAndType

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
  Arg : Nat -> Expr
  Tuple : List Nat -> Expr
  GetTupleElement : Nat -> Nat -> Expr

  ||| Apply a cached function to arguments.
  |||
  ||| @f The function pointer.
  ||| @xs The function arguments.
  Call : (f : Nat) -> (xs : List Nat) -> Expr

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
addFn : {arity : _} -> Vect arity ShapeAndType -> FnExpr arity -> State Env (Fn arity)
addFn params f = do
  MkEnv (nc, children) (next, env) <- get
  let (subEnv@(MkEnv (nc, _) (next, _)), params, result) = runState (MkEnv (nc, []) (next, [])) $ do
        xs <- traverse addArg params
        result <- applyN f xs
        pure (zip xs params, result)
  put (MkEnv (nc, children) (next, env))
  pure (MkFn params result subEnv)

  where
  addArg : ShapeAndType -> State Env Nat
  addArg st = do
    MkEnv children (next, env) <- get
    put (MkEnv children (S next, (next, Arg next) :: env))
    pure next

export
shareFn : {arity : _} -> Vect arity ShapeAndType -> FnExpr arity -> State Env Nat
shareFn params f = do
  fn <- addFn params f
  MkEnv (nc, comps) ops <- get
  put (MkEnv (S nc, (nc, (_ ** fn)) :: comps) ops)
  pure nc
