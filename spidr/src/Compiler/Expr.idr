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

public export
data Parameter : Type where
  MkParameter : Shape -> (0 dtype : Type) -> Primitive dtype => Parameter

%runElab derive "Parameter" [Show]

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
         (params : Vect arity (Nat, Parameter)) ->
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
  GetTupleElement : (index : Nat) -> Nat -> Expr
  MinValue : Primitive dtype => Expr
  MaxValue : Primitive dtype => Expr
  MinFiniteValue : Primitive dtype => Expr
  MaxFiniteValue : Primitive dtype => Expr
  Iota : Primitive dtype => (shape : Shape) -> (axis : Nat) -> Expr
  ConvertElementType : Primitive dtype => Nat -> Expr
  Reshape : (from, to : Shape) -> Nat -> Expr
  Slice : (starts, stops, strides : List Nat) -> Nat -> Expr
  DynamicSlice : (starts : List Nat) -> (sizes : List Nat) -> Nat -> Expr
  Concat : (axis : Nat) -> Nat -> Nat -> Expr
  Diag : Nat -> Expr
  Triangle : (lower : Bool) -> Nat -> Expr
  Transpose : (ordering : List Nat) -> Nat -> Expr
  Identity : Primitive dtype => (size : Nat) -> Expr
  Broadcast : Primitive dtype => (from, to : Shape) -> Nat -> Expr
  Map : Fn arity -> Vect arity Nat -> Shape -> Expr
  Reduce : Fn 2 -> (neutral : Nat) -> (axes : List Nat) -> Nat -> Expr
  Sort : Fn 2 -> (axis : Nat) -> (isStable : Bool) -> List Nat -> Expr
  Reverse : (axes : List Nat) -> Nat -> Expr
  BinaryElementwise : BinaryOp -> Nat -> Nat -> Expr
  UnaryElementwise : UnaryOp -> Nat -> Expr
  Argmin : Primitive out => Nat -> Nat -> Expr
  Argmax : Primitive out => Nat -> Nat -> Expr
  Select : Nat -> Nat -> Nat -> Expr
  Cond : (pred : Nat) -> (onTrue : Fn 1) -> (onTrueArg : Nat) ->
         (onFalse : Fn 1) -> (onFalseArg : Nat) -> Expr
  Dot : Nat -> Nat -> Expr
  DotGeneral : (lBatch, lContract, rBatch, rContract : List Nat) -> Nat -> Nat -> Expr
  Cholesky : Nat -> Expr
  TriangularSolve : Nat -> Nat -> (isLower : Bool) -> Expr
  UniformFloatingPoint : Nat -> Nat -> Nat -> Nat -> Shape -> Expr
  NormalFloatingPoint : Nat -> Nat -> Shape -> Expr

Show a => Interpolation (List a) where
  interpolate = show

showExpr : Nat -> Expr -> String

showEnv : Nat -> Env -> String
showEnv indent (MkEnv max env) = joinBy "\n" $ assert_total $ map fmt (reverse env)

  where

  fmt : (Nat, Expr) -> String
  fmt (n, x) =
    let sep = replicate (4 + length (show max) `minus` length (show n)) ' '
     in "\{replicate indent ' '}\{show n}\{sep}\{showExpr indent x}"

showFn : Nat -> Fn arity -> String
showFn indent (MkFn params result env) =
  """
  MkFn {parameters = \{show params}, root = \{show result}, env =
  \{showEnv (indent + 4) env}
  \{replicate (indent + 2) ' '}}
  """

export Show (Fn arity) where show = showFn 0

showExpr _      (FromLiteral {shape, dtype} x) = "FromLiteral \{shape} \{xlaIdentifier {dtype}}"
showExpr _      (Arg k) = "Arg \{k}"
showExpr _      (Tuple xs) = "Tuple \{xs}"
showExpr _      (GetTupleElement k xs) = "GetTupleElement {index = \{k}} \{xs}"
showExpr _      (MinValue {dtype}) = "MinValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr _      (MaxValue {dtype}) = "MaxValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr _      (MinFiniteValue {dtype}) = "MinFiniteValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr _      (MaxFiniteValue {dtype}) = "MaxFiniteValue {dtype = \{xlaIdentifier {dtype}}}"
showExpr _      (Iota {dtype} shape axis) =
  "Iota {shape = \{show shape}, dtype = \{xlaIdentifier {dtype}}, axis = \{axis}}"
showExpr _      (ConvertElementType {dtype} x) =
  "ConvertElementType {dtype = \{xlaIdentifier {dtype}}} \{x}"
showExpr _      (Reshape from to x) = "Reshape {from = \{from}, to = \{to}} \{x}"
showExpr _      (Slice starts stops strides x) =
  "Slice {starts = \{starts}, stops = \{stops}, strides = \{strides}} \{x}"
showExpr _      (DynamicSlice starts sizes x) = "Slice {starts = \{starts}, sizes = \{sizes} \{x}"
showExpr _      (Concat axis x y) = "Concat {axis = \{axis}} \{x} \{y}"
showExpr _      (Diag x) = "Diag \{x}"
showExpr _      (Triangle lower x) = "Triangle {lower = \{show lower}} \{x}"
showExpr _      (Transpose ordering x) = "Transpose {ordering = \{ordering}} \{x}"
showExpr _      (Identity {dtype} size) =
  "Identity {size = \{size}, dtype = \{xlaIdentifier {dtype}}}"
showExpr _      (Broadcast from to x) = "Broadcast {from = \{from}, to = \{to}} \{x}"
showExpr indent (Map f xs _) = "Map {f = \{showFn indent f}} \{show xs}"
showExpr indent (Reduce op neutral axes x) =
  "Reduce {op = \{showFn indent op}, identity = \{neutral}, axes = \{axes}} \{x}"
showExpr indent (Sort f axis _ xs) = "Sort {f = \{showFn indent f}, axis = \{axis}} \{xs}"
showExpr _      (Reverse axes x) = "Reverse \{axes} \{x}"
showExpr _      (BinaryElementwise op x y) = "\{show op} \{x} \{y}"
showExpr _      (UnaryElementwise op x) = "\{show op} \{x}"
showExpr _      (Argmin {out} x y) = "Argmin {outType = \{xlaIdentifier {dtype = out}}} \{x} \{y}"
showExpr _      (Argmax {out} x y) = "Argmax {outType = \{xlaIdentifier {dtype = out}}} \{x} \{y}"
showExpr _      (Select p t f) = "Select {predicate = \{p}, onTrue = \{t}, onFalse = \{f}}"
showExpr indent (Cond p ft t ff f) =
  "Cond {predicate = \{p}, onTrueFn = \{showFn indent ft}, onTrueArg = \{show t}," ++
    " onFalseFn = \{showFn indent ff}, onFalseArg = \{show f}}"
showExpr _      (Dot x y) = "Dot \{x} \{y}"
showExpr _      (DotGeneral lBatch lContract rBatch rContract x y) =
  "DotGeneral {lBatch = \{lBatch}, lContract = \{lContract}," ++
    " rBatch = \{rBatch}, rContract = \{rContract}} \{x} \{y}"
showExpr _      (Cholesky x) = "Cholesky \{x}"
showExpr _      (TriangularSolve x y isLower) =
  "TriangularSolve {isLower = \{show isLower}} \{x} \{y}"
showExpr _      (UniformFloatingPoint key initialState minval maxval shape) =
  "UniformFloatingPoint {key = \{key}, initialState = \{initialState}," ++
    " minval = \{minval}, maxval = \{maxval}, shape = \{shape}}"
showExpr _      (NormalFloatingPoint key initialState shape) =
  "NormalFloatingPoint {key = \{key}, initialState = \{initialState}, shape = \{shape}}"

public export 0
FnExpr : Nat -> Type
FnExpr 0 = State Env Nat
FnExpr (S k) = Nat -> FnExpr k

applyN : FnExpr arity -> Vect arity Nat -> State Env Nat
applyN f [] = f
applyN f (x :: xs) = applyN (f x) xs

export
addFn : {arity : _} -> Vect arity Parameter -> FnExpr arity -> State Env (Fn arity)
addFn params f = do
  MkEnv next env <- get
  let (subEnv@(MkEnv next _), params, result) = runState (MkEnv next []) $ do
        xs <- traverse addArg params
        result <- applyN f xs
        pure (zip xs params, result)
  put (MkEnv next env)
  pure (MkFn params result subEnv)

  where
  addArg : Parameter -> State Env Nat
  addArg st = do
    MkEnv next env <- get
    put (MkEnv (S next) ((next, Arg next) :: env))
    pure next
