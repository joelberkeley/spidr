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

import Control.Monad.State
import Data.Primitives.Interpolation
import Decidable.Equality

import Derive.Prelude
import Language.Reflection

import Compiler.LiteralRW
import Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util

%language ElabReflection

Show a => Interpolation (List a) where
  interpolate = show

public export
data Parameter : Type where
  MkParameter : Shape -> (0 dtype : Type) -> Primitive dtype => Parameter

Show Parameter where
  show (MkParameter shape dtype) = "\{shape} \{xlaIdentifier {dtype}}"

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
emptyFrom : Env -> Env
emptyFrom (MkEnv n _) = MkEnv n []

export
updateCounterFrom : Env -> State Env ()
updateCounterFrom (MkEnv n _) = do
  MkEnv _ xs <- get
  put $ MkEnv n xs

export
toList : Env -> List (Nat, Expr)
toList (MkEnv _ env) = reverse env

export
counter : Env -> Nat
counter (MkEnv c _) = c

public export
data Fn : Nat -> Type where

  ||| @arity The function arity.
  ||| @params The function parameter shapes and dtypes.
  ||| @result The function result.
  ||| @env Bindings within the function. Includes only nodes in this scope, not outer or inner scope.
  MkFn : {arity : _} ->
         (params : Vect arity (Nat, Parameter)) ->
         (result : Expr) ->
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
  Var : Nat -> Expr
  Tuple : List Expr -> Expr
  GetTupleElement : (index : Nat) -> Expr -> Expr
  MinValue : Primitive dtype => Expr
  MaxValue : Primitive dtype => Expr
  MinFiniteValue : Primitive dtype => Expr
  MaxFiniteValue : Primitive dtype => Expr
  Iota : Primitive dtype => (shape : Shape) -> (axis : Nat) -> Expr
  ConvertElementType : Primitive dtype => Expr -> Expr
  Reshape : (from, to : Shape) -> Expr -> Expr
  Slice : (starts, stops, strides : List Nat) -> Expr -> Expr
  DynamicSlice : (starts : List Expr) -> (sizes : List Nat) -> Expr -> Expr
  Concat : (axis : Nat) -> Expr -> Expr -> Expr
  Diag : Expr -> Expr
  Triangle : (lower : Bool) -> Expr -> Expr
  Transpose : (ordering : List Nat) -> Expr -> Expr
  Identity : Primitive dtype => (size : Nat) -> Expr
  Broadcast : Primitive dtype => (from, to : Shape) -> Expr -> Expr
  Map : Fn arity -> Vect arity Expr -> Shape -> Expr
  Reduce : Fn 2 -> (neutral : Expr) -> (axes : List Nat) -> Expr -> Expr
  Sort : Fn 2 -> (axis : Nat) -> (isStable : Bool) -> List Expr -> Expr
  Reverse : (axes : List Nat) -> Expr -> Expr
  BinaryElementwise : BinaryOp -> Expr -> Expr -> Expr
  UnaryElementwise : UnaryOp -> Expr -> Expr
  Argmax : Primitive out => (axis : Nat) -> Expr -> Expr
  Select : (predicate, onTrue, onFalse : Expr) -> Expr
  Cond : (pred : Expr) -> (onTrue : Fn 1) -> (onTrueArg : Expr) ->
         (onFalse : Fn 1) -> (onFalseArg : Expr) -> Expr
  Dot : Expr -> Expr -> Expr
  DotGeneral : (lBatch, lContract, rBatch, rContract : List Nat) -> Expr -> Expr -> Expr
  Cholesky : Expr -> Expr
  TriangularSolve : Expr -> Expr -> (isLower : Bool) -> Expr
  UniformFloatingPoint : (key, initialState, minval, maxval : Expr) -> (shape : Shape) -> Expr
  NormalFloatingPoint : (key, initialState : Expr) -> (shape : Shape) -> Expr

export
tag : Monad m => Expr -> StateT Env m Expr
tag expr = do
  MkEnv next env <- get
  put $ MkEnv (S next) ((next, expr) :: env)
  pure (Var next)

export
reserve : State Env Nat
reserve = do
  MkEnv next env <- get
  put $ MkEnv (S next) env
  pure next

covering
showExpr : Nat -> Expr -> String

covering
showExprList : Nat -> List Expr -> String
showExprList indent xs = "[" ++ joinBy ", " (toList $ map (showExpr indent) xs) ++ "]"

covering
showEnv : Nat -> Env -> String
showEnv indent (MkEnv max env) = joinBy "\n" $ assert_total $ map fmt (reverse env)

  where

  fmt : (Nat, Expr) -> String
  fmt (n, x) =
    let sep = replicate (4 + length (show max) `minus` length (show n)) ' '
     in "\{replicate indent ' '}\{show n}\{sep}\{showExpr indent x}"

covering
showFn : Nat -> Fn arity -> String
showFn indent (MkFn params result env@(MkEnv _ env')) =
  let init = "\{show params} => \{showExpr (indent + 2) result}" in
  case env' of
    [] => init
    _  => init ++ ", with vars {\n\{showEnv (indent + 4) env}\n\{replicate (indent + 2) ' '}}"

export Show (Fn arity) where show = assert_total $ showFn 0

showExpr indent (FromLiteral {shape, dtype} x) = "Lit \{shape} \{xlaIdentifier {dtype}}"
showExpr indent (Var k) = "Var \{k}"
showExpr indent (Tuple xs) = "Tuple \{showExprList indent xs}"
showExpr indent (GetTupleElement k x) = "GetTupleElement {index = \{k}} (\{showExpr indent x})"
showExpr indent (MinValue {dtype}) = "MinValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr indent (MaxValue {dtype}) = "MaxValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr indent (MinFiniteValue {dtype}) = "MinFiniteValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr indent (MaxFiniteValue {dtype}) = "MaxFiniteValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr indent (Iota {dtype} shape axis) =
  "Iota {shape = \{show shape}, dtype = \{xlaIdentifier {dtype}}, axis = \{axis}}"
showExpr indent (ConvertElementType {dtype} x) =
  "ConvertElementType {dtype = \{xlaIdentifier {dtype}}} (\{showExpr indent x})"
showExpr indent (Reshape from to x) = "Reshape {from = \{from}, to = \{to}} (\{showExpr indent x})"
showExpr indent (Slice starts stops strides x) =
  "Slice {starts = \{starts}, stops = \{stops}, strides = \{strides}} (\{showExpr indent x})"
showExpr indent (DynamicSlice starts sizes x) =
  "DynamicSlice {starts = \{showExprList indent starts}, sizes = \{sizes}} (\{showExpr indent x})"
showExpr indent (Concat axis x y) =
  "Concat {axis = \{axis}} (\{showExpr indent x}) (\{showExpr indent y})"
showExpr indent (Diag x) = "Diag (\{showExpr indent x})"
showExpr indent (Triangle lower x) = "Triangle {lower = \{show lower}} (\{showExpr indent x})"
showExpr indent (Transpose ordering x) = "Transpose {ordering = \{ordering}} (\{showExpr indent x})"
showExpr indent (Identity {dtype} size) =
  "Identity {size = \{size}, dtype = \{xlaIdentifier {dtype}}}"
showExpr indent (Broadcast from to x) =
  "Broadcast {from = \{from}, to = \{to}} (\{showExpr indent x})"
showExpr indent (Map f xs _) = "Map {f = \{showFn indent f}} \{showExprList indent $ toList xs}"
showExpr indent (Reduce op neutral axes x) =
  "Reduce {op = \{showFn indent op}, identity = \{showExpr indent neutral}," ++
    " axes = \{axes}} (\{showExpr indent x})"
showExpr indent (Sort f axis _ xs) =
  "Sort {f = \{showFn indent f}, axis = \{axis}} \{showExprList indent xs}"
showExpr indent (Reverse axes x) = "Reverse \{axes} (\{showExpr indent x})"
showExpr indent (BinaryElementwise op x y) =
  "\{show op} (\{showExpr indent x}) (\{showExpr indent y})"
showExpr indent (UnaryElementwise op x) = "\{show op} (\{showExpr indent x})"
showExpr indent (Argmax {out} axis x) =
  "Argmax {outType = \{xlaIdentifier {dtype = out}}} \{axis} (\{showExpr indent x})"
showExpr indent (Select p t f) =
  "Select {predicate = \{showExpr indent p}, onTrue = \{showExpr indent t}," ++
    " onFalse = \{showExpr indent f}}"
showExpr indent (Cond p ft t ff f) =
  "Cond {predicate = \{showExpr indent p}, onTrueFn = \{showFn indent ft}," ++
    " onTrueArg = \{showExpr indent t}, onFalseFn = \{showFn indent ff}," ++
    " onFalseArg = \{showExpr indent f}}"
showExpr indent (Dot x y) = "Dot (\{showExpr indent x}) (\{showExpr indent y})"
showExpr indent (DotGeneral lBatch lContract rBatch rContract x y) =
  "DotGeneral {lBatch = \{lBatch}, lContract = \{lContract}," ++
    " rBatch = \{rBatch}, rContract = \{rContract}} (\{showExpr indent x}) (\{showExpr indent y})"
showExpr indent (Cholesky x) = "Cholesky (\{showExpr indent x})"
showExpr indent (TriangularSolve x y isLower) =
  "TriangularSolve {isLower = \{show isLower}} (\{showExpr indent x}) (\{showExpr indent y})"
showExpr indent (UniformFloatingPoint key initialState minval maxval shape) =
  "UniformFloatingPoint {key = \{showExpr indent key}," ++
    " initialState = \{showExpr indent initialState}," ++
    " minval = \{showExpr indent minval}, maxval = \{showExpr indent maxval}, shape = \{shape}}"
showExpr indent (NormalFloatingPoint key initialState shape) =
  "NormalFloatingPoint {key = \{showExpr indent key}," ++
    " initialState = \{showExpr indent initialState}, shape = \{shape}}"
