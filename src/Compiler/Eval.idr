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
import Data.SortedMap
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
import Util.Hashable

%hide Util.List.All.map

data Err = IndexErr String

Computation : Type -> Type
Computation = StateT (XlaBuilder, List XlaOp) IO

build : HasIO io => String -> Computation XlaOp -> io XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  (_, root) <- liftIO $ runStateT (builder, []) x
  build builder root

buildWithSubBuilder :
  String -> List (Computation XlaOp) -> Computation XlaOp -> Computation XlaComputation
buildWithSubBuilder computationName computationArguments computationResult = do
  (builder, _) <- get
  subBuilder <- createSubBuilder builder computationName
  (_, cache) <- liftIO $ execStateT (subBuilder, empty) (sequence_ computationArguments)
  (_, root) <- liftIO $ runStateT (subBuilder, cache) computationResult
  build subBuilder root

index : Nat -> List a -> Either Err a

covering
enqueue : List XlaOp -> Expr -> Either Err (Computation XlaOp)
enqueue _ (FromLiteral {dtype} lit) = pure $ do
  (builder, _) <- get
  literal <- write {dtype} lit 
  constantLiteral builder literal
enqueue _ (Parameter {dtype} position shape name) = pure $ do
  (builder, _) <- get
  xlaShape <- mkShape {dtype} shape
  parameter builder position xlaShape name
enqueue ops (Tuple xs) = do
  tupleOps <- traverse (flip index ops) xs
  pure $ do
    (builder, _) <- get
    tuple builder tupleOps
enqueue ops (GetTupleElement idx x) = do
  op <- index x ops
  pure $ do getTupleElement op idx
enqueue _ (MinValue {dtype}) = pure $ do
  (builder, _) <- get
  minValue {dtype} builder
enqueue _ (MaxValue {dtype}) = pure $ do
  (builder, _) <- get
  maxValue {dtype} builder
enqueue _ (MinFiniteValue {dtype}) = pure $ do
  (builder, _) <- get
  minFiniteValue {dtype} builder
enqueue _ (MaxFiniteValue {dtype}) = pure $ do
  (builder, _) <- get
  maxFiniteValue {dtype} builder
-- todo test dtypes here, is this a bug?
enqueue ops (ConvertElementType x) = do
  op <- index x ops
  pure $ do convertElementType {dtype = F64} op
enqueue ops (Reshape from to x) = do
  op <- index x ops
  pure $ reshape op (range $ length from) to
enqueue ops (Slice starts stops strides x) = do
  op <- index x ops
  pure $ slice op starts stops strides
enqueue ops (DynamicSlice starts sizes x) = do
  starts <- traverse (flip index ops) starts
  op <- index x ops
  pure $ dynamicSlice op starts sizes
enqueue ops (Concat axis x x') = do
  op <- index x ops
  op' <- index x' ops
  pure $ do
    (builder, _) <- get
    concatInDim builder [op, op'] (cast axis)
enqueue ops (Diag x) = do
  op <- index x ops
  pure $ getMatrixDiagonal op
enqueue ops (Triangle tri x) = do
  op <- index x ops
  pure $ triangle op tri
enqueue ops (Transpose ordering x) = do
  op <- index x ops
  pure $ transpose op ordering
enqueue _ (Identity {dtype} n) = let n = cast n in Right $ do
  (builder, _) <- get
  identityMatrix {dtype} builder n n
enqueue ops (Broadcast {dtype} from to x) =
  if elem 0 to && from /= to
  then Right $ do
    (builder, _) <- get
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
   let broadcastDims = map (+ length to `minus` length from) $ range $ length from
    in do
       op <- index x ops
       pure $ broadcastInDim op to broadcastDims
enqueue ops (Map (MkFn {arity} exprParams exprf) exprs dims) = do
  computation <- buildWithSubBuilder "computation" (map enqueue $ toList exprParams) (enqueue exprf)
  (builder, _) <- get
  map builder !(traverse enqueue $ toList exprs) computation dims 
enqueue ops (Reduce (MkFn [p0, p1] exprf) neutral axes expr) = do
  computation <- buildWithSubBuilder "computation" [(enqueue p0), (enqueue p1)] (enqueue exprf) 
  reduce !(enqueue expr) !(enqueue neutral) computation axes
enqueue ops (Sort (MkFn [p0, p1] exprComp) axis isStable exprs) = do
  comparator <- buildWithSubBuilder "comparator" [(enqueue p0), (enqueue p1)] (enqueue exprComp)
  sort !(traverse enqueue exprs) comparator axis isStable 
enqueue ops (Reverse axes expr) = rev !(enqueue expr) axes
enqueue ops (Eq l r) = eq !(enqueue l) !(enqueue r)
enqueue ops (Ne l r) = ne !(enqueue l) !(enqueue r)
enqueue ops (Add l r) = add !(enqueue l) !(enqueue r)
enqueue ops (Sub l r) = sub !(enqueue l) !(enqueue r)
enqueue ops (Mul l r) = mul !(enqueue l) !(enqueue r)
enqueue ops (Div l r) = div !(enqueue l) !(enqueue r)
enqueue ops (Pow l r) = pow !(enqueue l) !(enqueue r)
enqueue ops (Lt l r) = lt !(enqueue l) !(enqueue r)
enqueue ops (Gt l r) = gt !(enqueue l) !(enqueue r)
enqueue ops (Le l r) = le !(enqueue l) !(enqueue r)
enqueue ops (Ge l r) = ge !(enqueue l) !(enqueue r)
enqueue ops (And l r) = and !(enqueue l) !(enqueue r)
enqueue ops (Or l r) = or !(enqueue l) !(enqueue r)
enqueue ops (Min l r) = min !(enqueue l) !(enqueue r)
enqueue ops (Max l r) = max !(enqueue l) !(enqueue r)
enqueue ops (Not expr) = not !(enqueue expr)
enqueue ops (Neg expr) = neg !(enqueue expr)
enqueue ops (Reciprocal expr) = reciprocal !(enqueue expr)
enqueue ops (Abs expr) = abs !(enqueue expr)
enqueue ops (Ceil expr) = ceil !(enqueue expr)
enqueue ops (Floor expr) = floor !(enqueue expr)
enqueue ops (Exp expr) = exp !(enqueue expr)
enqueue ops (Log expr) = log !(enqueue expr)
enqueue ops (Logistic expr) = logistic !(enqueue expr)
enqueue ops (Erf expr) = erf !(enqueue expr)
enqueue ops (Square expr) = square !(enqueue expr)
enqueue ops (Sqrt expr) = sqrt !(enqueue expr)
enqueue ops (Sin expr) = sin !(enqueue expr)
enqueue ops (Cos expr) = cos !(enqueue expr)
enqueue ops (Tan expr) = tan !(enqueue expr)
enqueue ops (Asin expr) = asin !(enqueue expr)
enqueue ops (Acos expr) = acos !(enqueue expr)
enqueue ops (Atan expr) = atan !(enqueue expr)
enqueue ops (Sinh expr) = sinh !(enqueue expr)
enqueue ops (Cosh expr) = cosh !(enqueue expr)
enqueue ops (Tanh expr) = tanh !(enqueue expr)
enqueue ops (Asinh expr) = asinh !(enqueue expr)
enqueue ops (Acosh expr) = acosh !(enqueue expr)
enqueue ops (Atanh expr) = atanh !(enqueue expr)
enqueue ops (Argmin {out} axis expr) = argMin {outputType=out} !(enqueue expr) axis
enqueue ops (Argmax {out} axis expr) = argMax {outputType=out} !(enqueue expr) axis
enqueue ops (Select pred true false) =
  select !(enqueue pred) !(enqueue true) !(enqueue false)
enqueue ops (Cond pred (MkFn [pt] exprTrue) true (MkFn [pf] exprFalse) false) = do
  trueComp <- buildWithSubBuilder "truthy computation" [enqueue pt] (enqueue exprTrue)
  falseComp <- buildWithSubBuilder "falsy computation" [enqueue pf] (enqueue exprFalse)
  conditional !(enqueue pred) !(enqueue true) trueComp !(enqueue false) falseComp
enqueue ops (Dot l r) = dot !(enqueue l) !(enqueue r)
enqueue ops (Cholesky expr) = cholesky !(enqueue expr) True
enqueue ops (TriangularSolve a b lower) =
  triangularSolve !(enqueue a) !(enqueue b) True lower False NoTranspose
enqueue ops (UniformFloatingPoint key initialState minval maxval shape) = do
  rngOutput <- uniformFloatingPointDistribution
    !(enqueue key)
    !(enqueue initialState)
    ThreeFry
    !(enqueue minval)
    !(enqueue maxval)
    !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]
enqueue ops (NormalFloatingPoint key initialState shape) = do
  rngOutput <- normalFloatingPointDistribution
    !(enqueue key) !(enqueue initialState) ThreeFry !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]

export
toString : Expr -> String
toString expr = unsafePerformIO $ do
  builder <- mkXlaBuilder "toString"
  let comp = assert_total (enqueue expr)
  ((builder, _), xlaOp) <- runStateT (builder, empty) comp
  pure $ opToString builder xlaOp

export
run : PrimitiveRW dtype a => Expr -> {shape : _} -> Literal shape a
run expr = unsafePerformIO $ do
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  computation <- build "" (assert_total $ enqueue expr)
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
