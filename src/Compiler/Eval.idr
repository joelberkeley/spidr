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

import Control.Monad.State
import Data.List
import Data.List.Elem
import Data.SortedMap
import Decidable.Equality

import Data.Hashable

import Compiler.Expr
import Compiler.LiteralRW
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

Cache : Type
Cache = SortedMap Bits64 (List (Expr, XlaOp))

Computation : Type -> Type
Computation = StateT (XlaBuilder, Cache) IO

cached : Expr -> Computation XlaOp -> Computation XlaOp
cached graph xs = let graphHash = hash graph in do
  (_, cache) <- get
  case lookup graphHash cache of
    Just candidates => case find (\(graph', _) => graph' == graph) candidates of
      Just (_, op) => pure op
      Nothing => runOp xs graphHash graph candidates
    Nothing => runOp xs graphHash graph []

  where
  runOp : Computation XlaOp -> Bits64 -> Expr -> List (Expr, XlaOp) -> Computation XlaOp
  runOp xs key graph graphOps = do
    op <- xs
    (builder, cache) <- get
    put (builder, insert key ((graph, op) :: graphOps) cache)
    pure op

build : HasIO io => String -> Computation XlaOp -> io XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  (_, root) <- liftIO $ runStateT (builder, empty) x
  build builder root

buildWithSubBuilder :
  String -> List (Computation XlaOp) -> Computation XlaOp -> Computation XlaComputation
buildWithSubBuilder computationName computationArguments computationResult = do
  (builder, _) <- get
  subBuilder <- createSubBuilder builder computationName
  (_, cache) <- liftIO $ execStateT (subBuilder, empty) (sequence_ computationArguments)
  (_, root) <- liftIO $ runStateT (subBuilder, cache) computationResult
  build subBuilder root

covering
enqueue : Expr -> Computation XlaOp
enqueue e@(FromLiteral {dtype} lit) = cached e $ do
  (builder, _) <- get
  literal <- write {dtype} lit 
  constantLiteral builder literal
enqueue e@(Parameter {dtype} position shape name) = cached e $ do
  (builder, _) <- get
  xlaShape <- mkShape {dtype} shape
  parameter builder position xlaShape name
enqueue e@(Tuple exprs) = cached e $ do
  (builder, _) <- get
  tuple builder !(traverse enqueue exprs)
enqueue e@(GetTupleElement idx expr) = cached e $ getTupleElement !(enqueue expr) idx
enqueue e@(MinValue {dtype}) = cached e $ do
  (builder, _) <- get
  minValue {dtype} builder
enqueue e@(MaxValue {dtype}) = cached e $ do
  (builder, _) <- get
  maxValue {dtype} builder
enqueue e@(MinFiniteValue {dtype}) = cached e $ do
  (builder, _) <- get
  minFiniteValue {dtype} builder
enqueue e@(MaxFiniteValue {dtype}) = cached e $ do
  (builder, _) <- get
  maxFiniteValue {dtype} builder
enqueue e@(ConvertElementType expr) = cached e $ convertElementType {dtype=F64} !(enqueue expr)
enqueue e@(Reshape from to expr) = cached e $ reshape !(enqueue expr) (range $ length from) to
enqueue e@(Slice starts stops strides expr) = cached e $ slice !(enqueue expr) starts stops strides 
enqueue e@(DynamicSlice starts sizes expr) =
  cached e $ dynamicSlice !(enqueue expr) !(traverse enqueue starts) sizes
enqueue e@(Concat axis expr expr') = cached e $ do
  (builder, _) <- get
  concatInDim builder [!(enqueue expr), !(enqueue expr')] (cast axis)
enqueue e@(Diag expr) = cached e $ getMatrixDiagonal !(enqueue expr)
enqueue e@(Triangle tri expr) = cached e $ triangle !(enqueue expr) tri
enqueue e@(Transpose ordering expr) = cached e $ transpose !(enqueue expr) ordering
enqueue e@(Identity {dtype} n) = cached e $ let n = cast n in do
  (builder, _) <- get
  identityMatrix {dtype} builder n n
enqueue e@(Broadcast {dtype} from to expr) = cached e $
  if elem 0 to && from /= to
  then do
    (builder, _) <- get
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
    let broadcastDims = map (+ length to `minus` length from) $ range $ length from
     in broadcastInDim !(enqueue expr) to broadcastDims
enqueue e@(Map (MkFn {arity} exprParams exprf) exprs dims) = cached e $ do
  computation <- buildWithSubBuilder "computation" (map enqueue $ toList exprParams) (enqueue exprf)
  (builder, _) <- get
  map builder !(traverse enqueue $ toList exprs) computation dims 
enqueue e@(Reduce (MkFn [p0, p1] exprf) neutral axes expr) = cached e $ do
  computation <- buildWithSubBuilder "computation" [(enqueue p0), (enqueue p1)] (enqueue exprf) 
  reduce !(enqueue expr) !(enqueue neutral) computation axes
enqueue e@(Sort (MkFn [p0, p1] exprComp) axis isStable exprs) = cached e $ do
  comparator <- buildWithSubBuilder "comparator" [(enqueue p0), (enqueue p1)] (enqueue exprComp)
  sort !(traverse enqueue exprs) comparator axis isStable 
enqueue e@(Reverse axes expr) = cached e $ rev !(enqueue expr) axes
enqueue e@(Eq l r) = cached e $ eq !(enqueue l) !(enqueue r)
enqueue e@(Ne l r) = cached e $ ne !(enqueue l) !(enqueue r)
enqueue e@(Add l r) = cached e $ add !(enqueue l) !(enqueue r)
enqueue e@(Sub l r) = cached e $ sub !(enqueue l) !(enqueue r)
enqueue e@(Mul l r) = cached e $ mul !(enqueue l) !(enqueue r)
enqueue e@(Div l r) = cached e $ div !(enqueue l) !(enqueue r)
enqueue e@(Pow l r) = cached e $ pow !(enqueue l) !(enqueue r)
enqueue e@(Lt l r) = cached e $ lt !(enqueue l) !(enqueue r)
enqueue e@(Gt l r) = cached e $ gt !(enqueue l) !(enqueue r)
enqueue e@(Le l r) = cached e $ le !(enqueue l) !(enqueue r)
enqueue e@(Ge l r) = cached e $ ge !(enqueue l) !(enqueue r)
enqueue e@(And l r) = cached e $ and !(enqueue l) !(enqueue r)
enqueue e@(Or l r) = cached e $ or !(enqueue l) !(enqueue r)
enqueue e@(Min l r) = cached e $ min !(enqueue l) !(enqueue r)
enqueue e@(Max l r) = cached e $ max !(enqueue l) !(enqueue r)
enqueue e@(Not expr) = cached e $ not !(enqueue expr)
enqueue e@(Neg expr) = cached e $ neg !(enqueue expr)
enqueue e@(Reciprocal expr) = cached e $ reciprocal !(enqueue expr)
enqueue e@(Abs expr) = cached e $ abs !(enqueue expr)
enqueue e@(Ceil expr) = cached e $ ceil !(enqueue expr)
enqueue e@(Floor expr) = cached e $ floor !(enqueue expr)
enqueue e@(Exp expr) = cached e $ exp !(enqueue expr)
enqueue e@(Log expr) = cached e $ log !(enqueue expr)
enqueue e@(Logistic expr) = cached e $ logistic !(enqueue expr)
enqueue e@(Erf expr) = cached e $ erf !(enqueue expr)
enqueue e@(Square expr) = cached e $ square !(enqueue expr)
enqueue e@(Sqrt expr) = cached e $ sqrt !(enqueue expr)
enqueue e@(Sin expr) = cached e $ sin !(enqueue expr)
enqueue e@(Cos expr) = cached e $ cos !(enqueue expr)
enqueue e@(Tan expr) = cached e $ tan !(enqueue expr)
enqueue e@(Asin expr) = cached e $ asin !(enqueue expr)
enqueue e@(Acos expr) = cached e $ acos !(enqueue expr)
enqueue e@(Atan expr) = cached e $ atan !(enqueue expr)
enqueue e@(Sinh expr) = cached e $ sinh !(enqueue expr)
enqueue e@(Cosh expr) = cached e $ cosh !(enqueue expr)
enqueue e@(Tanh expr) = cached e $ tanh !(enqueue expr)
enqueue e@(Asinh expr) = cached e $ asinh !(enqueue expr)
enqueue e@(Acosh expr) = cached e $ acosh !(enqueue expr)
enqueue e@(Atanh expr) = cached e $ atanh !(enqueue expr)
enqueue e@(Select pred true false) = cached e $ select !(enqueue pred) !(enqueue true) !(enqueue false)
enqueue e@(Cond pred (MkFn [pt] exprTrue) true (MkFn [pf] exprFalse) false) = cached e $ do
  trueComp <- buildWithSubBuilder "truthy computation" [enqueue pt] (enqueue exprTrue)
  falseComp <- buildWithSubBuilder "falsy computation" [enqueue pf] (enqueue exprFalse)
  conditional !(enqueue pred) !(enqueue true) trueComp !(enqueue false) falseComp
enqueue e@(Dot l r) = cached e $ dot !(enqueue l) !(enqueue r)
enqueue e@(Cholesky expr) = cached e $ cholesky !(enqueue expr) True
enqueue e@(TriangularSolve a b lower) =
  cached e $ triangularSolve !(enqueue a) !(enqueue b) True lower False NoTranspose
enqueue e@(UniformFloatingPoint key initialState minval maxval shape) = cached e $ do
  rngOutput <- uniformFloatingPointDistribution
    !(enqueue key)
    !(enqueue initialState)
    ThreeFry
    !(enqueue minval)
    !(enqueue maxval)
    !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]
enqueue e@(NormalFloatingPoint key initialState shape) = cached e $ do
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
