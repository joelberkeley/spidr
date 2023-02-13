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
module Compiler.Eval

import Control.Monad.Error.Either
import Control.Monad.Maybe
import Control.Monad.State
import Data.List
import Data.List.Elem
import public Data.SortedMap
import Decidable.Equality

import Compiler.Expr
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Arithmetic
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Constants
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Math
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Matrix
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaComputation
import Compiler.Xla.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.TensorFlow.Compiler.Xla.Service.PlatformUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.ClientLibrary
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.LocalClient
import Compiler.Xla.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.Xla.TensorFlow.Core.Platform.Status
import Compiler.Xla.TensorFlow.StreamExecutor.Platform

import Literal
import Primitive
import Types
import Util

%hide Util.List.All.map

data Err = IndexErr String

Computation : Type -> Type
Computation = StateT (XlaBuilder, SortedMap Nat XlaOp) (EitherT Err IO)

build : String -> Computation XlaOp -> EitherT Err IO XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  root <- evalStateT (builder, empty) x
  XlaBuilder.build builder root

buildWithSubBuilder :
  XlaBuilder ->
  String ->
  List (Computation XlaOp) ->
  Computation XlaOp ->
  EitherT Err IO XlaComputation
buildWithSubBuilder builder computationName computationArguments computationResult = do
  subBuilder <- createSubBuilder builder computationName
  (_, cache) <- liftIO $ execStateT (subBuilder, empty) (pure $ sequence_ computationArguments)
  root <- evalStateT (subBuilder, cache) computationResult
  XlaBuilder.build subBuilder root

lookup : Nat -> SortedMap Nat a -> Computation a
lookup n xs =
  case lookup n xs of
       Nothing =>
         lift $ left (IndexErr "Tried to look up value at index \{show n} but none was found.")
       Just x => pure x

covering
enqueue : XlaBuilder -> SortedMap Nat XlaOp -> Expr -> Computation XlaOp
enqueue builder _ (FromLiteral {dtype} lit) = do
  literal <- write {dtype} lit
  constantLiteral builder literal
enqueue builder _ (Parameter {dtype} position shape name) = do
  xlaShape <- mkShape {dtype} shape
  parameter builder position xlaShape name
enqueue builder ops (Tuple xs) = do
  tupleOps <- traverse (flip lookup ops) xs
  tuple builder tupleOps
enqueue builder ops (GetTupleElement idx x) = getTupleElement !(lookup x ops) idx
enqueue builder _ (MinValue {dtype}) = minValue {dtype} builder
enqueue builder _ (MaxValue {dtype}) = maxValue {dtype} builder
enqueue builder _ (MinFiniteValue {dtype}) = minFiniteValue {dtype} builder
enqueue builder _ (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} builder
-- todo test dtypes here, is this a bug?
enqueue _ ops (ConvertElementType x) = convertElementType {dtype = F64} !(lookup x ops)
enqueue _ ops (Reshape from to x) = reshape !(lookup x ops) (range $ length from) to
enqueue _ ops (Slice starts stops strides x) = slice !(lookup x ops) starts stops strides
enqueue _ ops (DynamicSlice starts sizes x) =
  dynamicSlice !(lookup x ops) !(traverse (flip lookup ops) starts) sizes
enqueue builder ops (Concat axis x y) = do
  concatInDim builder [!(lookup x ops), !(lookup y ops)] (cast axis)
enqueue _ ops (Diag x) = getMatrixDiagonal !(lookup x ops)
enqueue _ ops (Triangle tri x) = triangle !(lookup x ops) tri
enqueue _ ops (Transpose ordering x) = transpose !(lookup x ops) ordering
enqueue builder _ (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} builder n n
enqueue builder ops (Broadcast {dtype} from to x) =
  if elem 0 to && from /= to
  then do
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
   let broadcastDims = map (+ length to `minus` length from) $ range $ length from
    in broadcastInDim !(lookup x ops) to broadcastDims
enqueue builder ops (Map (MkFn {arity} shapesAndTypes result) xs dims) = ?map -- do
  {-
  computation <- buildWithSubBuilder "computation" (map enqueue $ toList exprParams) (enqueue exprf)
  (builder, _) <- get
  map builder !(traverse enqueue $ toList exprs) computation dims 
  -}
enqueue builder ops (Reduce (MkFn [p0, p1] exprf) neutral axes expr) = ?reduce -- do
  {-
  computation <- buildWithSubBuilder "computation" [(enqueue p0), (enqueue p1)] (enqueue exprf) 
  reduce !(enqueue expr) !(enqueue neutral) computation axes
  -}
enqueue builder ops (Sort (MkFn [p0, p1] exprComp) axis isStable exprs) = ?sort -- do
  {-
  comparator <- buildWithSubBuilder "comparator" [(enqueue p0), (enqueue p1)] (enqueue exprComp)
  sort !(traverse enqueue exprs) comparator axis isStable 
  -}
enqueue _ ops (Reverse axes x) = rev !(lookup x ops) axes
enqueue _ ops (BinaryElementwise f x y) = toXla f !(lookup x ops) !(lookup y ops)
  where
  toXla : BinaryOp -> HasIO io => XlaOp -> XlaOp -> io XlaOp
  toXla Eq = eq
  toXla Ne = ne
  toXla Add = add
  toXla Sub = sub
  toXla Mul = mul
  toXla Div = div
  toXla Pow = pow
  toXla Lt = lt
  toXla Gt = gt
  toXla Le = le
  toXla Ge = ge
  toXla And = and
  toXla Or = or
  toXla Min = min
  toXla Max = max
enqueue _ ops (UnaryElementwise f x) = toXla f !(lookup x ops)
  where
  toXla : UnaryOp -> HasIO io => XlaOp -> io XlaOp
  toXla Not = not
  toXla Neg = neg
  toXla Reciprocal = reciprocal
  toXla Ceil = ceil
  toXla Floor = floor
  toXla Abs = abs
  toXla Log = log
  toXla Exp = exp
  toXla Logistic = logistic
  toXla Erf = erf
  toXla Square = square
  toXla Sqrt = sqrt
  toXla Sin = sin
  toXla Cos = cos
  toXla Tan = tan
  toXla Asin = asin
  toXla Acos = acos
  toXla Atan = atan
  toXla Sinh = sinh
  toXla Cosh = cosh
  toXla Tanh = tanh
  toXla Asinh = asinh
  toXla Acosh = acosh
  toXla Atanh = atanh
enqueue _ ops (Argmin {out} axis x) = argMin {outputType=out} !(lookup x ops) axis
enqueue _ ops (Argmax {out} axis x) = argMax {outputType=out} !(lookup x ops) axis
enqueue _ ops (Select pred true false) =
  select !(lookup pred ops) !(lookup true ops) !(lookup false ops)
enqueue builder ops (Cond pred (MkFn [pt] exprTrue) true (MkFn [pf] exprFalse) false) = ?cond -- do
{-
  trueComp <- buildWithSubBuilder "truthy computation" [enqueue pt] (enqueue exprTrue)
  falseComp <- buildWithSubBuilder "falsy computation" [enqueue pf] (enqueue exprFalse)
  conditional !(enqueue pred) !(enqueue true) trueComp !(enqueue false) falseComp
-}
enqueue _ ops (Dot l r) = dot !(lookup l ops) !(lookup r ops)
enqueue _ ops (Cholesky x) = cholesky !(lookup x ops) True
enqueue _ ops (TriangularSolve a b lower) =
  triangularSolve !(lookup a ops) !(lookup b ops) True lower False NoTranspose
enqueue builder ops (UniformFloatingPoint key initialState minval maxval shape) = ?ufp -- do
  {-
  rngOutput <- uniformFloatingPointDistribution
    !(enqueue key)
    !(enqueue initialState)
    ThreeFry
    !(enqueue minval)
    !(enqueue maxval)
    !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]
1  -}
enqueue builder ops (NormalFloatingPoint key initialState shape) = ?nfp -- do
  {-
  rngOutput <- normalFloatingPointDistribution
    !(enqueue key) !(enqueue initialState) ThreeFry !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]
  -}

interpret : Nat -> SortedMap Nat Expr -> Computation XlaOp
interpret root xs = do
  traverse_ processExpr (toList xs)
  (_, ops) <- get
  lookup root ops

  where
  processExpr : (Nat, Expr) -> Computation ()
  processExpr (n, expr) = do
    (builder, ops) <- get
    op <- enqueue builder ops expr
    put (builder, insert n op ops)

export
toString : Nat -> SortedMap Nat Expr -> EitherT Err IO String
toString rootIndex exprs = do
  builder <- mkXlaBuilder "toString"
  xlaOp <- evalStateT (builder, empty) (interpret rootIndex exprs)
  pure $ opToString builder xlaOp

export
run : PrimitiveRW dtype a => Nat -> SortedMap Nat Expr -> {shape : _} -> EitherT Err IO (Literal shape a)
run rootIndex exprs = do
  computation <- build "" (interpret rootIndex exprs)
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)

