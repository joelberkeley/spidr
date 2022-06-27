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

public export
data CachingBuilder : Type where
  -- can we can now separate the builder and cache?
  MkCachingBuilder : XlaBuilder -> SortedMap Bits64 (List (Expr, XlaOp)) -> CachingBuilder

public export
Computation : Type -> Type
Computation = StateT CachingBuilder IO

export
cached : Expr -> Computation XlaOp -> Computation XlaOp
cached graph xs = let graphHash = hash graph in do
  builder <- get
  case cacheLookup builder graphHash of
    Just candidates => case find (\(graph', _) => graph' == graph) candidates of
      Just (_, op) => pure op
      Nothing => runOp xs graphHash graph candidates
    Nothing => runOp xs graphHash graph []

  where
  cacheUpdate : CachingBuilder -> Bits64 -> List (Expr, XlaOp) -> CachingBuilder
  cacheUpdate (MkCachingBuilder builder cache) key graphOps =
    MkCachingBuilder builder (insert key graphOps cache)

  cacheLookup : CachingBuilder -> Bits64 -> Maybe (List (Expr, XlaOp))
  cacheLookup (MkCachingBuilder _ cache) key = lookup key cache

  runOp : Computation XlaOp -> Bits64 -> Expr -> List (Expr, XlaOp) -> Computation XlaOp
  runOp xs key graph graphOps = do
    op <- xs
    builder <- get
    put (cacheUpdate builder key ((graph, op) :: graphOps))
    pure op

export
build : HasIO io => String -> Computation XlaOp -> io XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  (MkCachingBuilder builder _, root) <- liftIO $ runStateT (MkCachingBuilder builder empty) x
  build builder root

export
buildWithSubBuilder :
  String -> List (Computation XlaOp) -> Computation XlaOp -> Computation XlaComputation
buildWithSubBuilder computationName computationArguments computationResult = do
  MkCachingBuilder builder _ <- get
  subBuilder <- createSubBuilder builder computationName
  let cachingSubBuilder = MkCachingBuilder subBuilder empty
  cachingSubBuilder <- liftIO $ execStateT cachingSubBuilder (sequence_ computationArguments)
  (MkCachingBuilder subBuilder _, root) <- liftIO $ runStateT cachingSubBuilder computationResult
  build subBuilder root

export
opToString : Computation XlaOp -> String
opToString x = unsafePerformIO $ do
  builder <- mkXlaBuilder "toString"
  (MkCachingBuilder builder _, xlaOp) <- runStateT (MkCachingBuilder builder empty) x
  pure $ opToString builder xlaOp

export
parameter : Primitive dtype => Nat -> Types.Shape -> String -> Computation XlaOp
parameter position shape name = do
  MkCachingBuilder builder _ <- get
  xlaShape <- mkShape {dtype} shape
  parameter builder position xlaShape name

export covering
eval : Expr -> Computation XlaOp
eval e@(FromLiteral {dtype} lit) = cached e $ do
  MkCachingBuilder builder _ <- get
  literal <- write {dtype} lit 
  constantLiteral builder literal
eval e@(Parameter {dtype} position shape name) = cached e $ parameter {dtype} position shape name
eval e@(MinFiniteValue {dtype}) = cached e $ do
  MkCachingBuilder builder _ <- get
  minFiniteValue {dtype} builder
eval e@(MaxFiniteValue {dtype}) = cached e $ do
  MkCachingBuilder builder _ <- get
  maxFiniteValue {dtype} builder
eval e@(ConvertElementType expr) = cached e $ convertElementType {dtype=F64} !(eval expr)
eval e@(Reshape from to expr) = cached e $ reshape !(eval expr) (range $ length from) to
eval e@(Slice starts stops strides expr) = cached e $ slice !(eval expr) starts stops strides 
eval e@(Concat axis expr expr') = cached e $ do
  MkCachingBuilder builder _ <- get
  concatInDim builder [!(eval expr), !(eval expr')] (cast axis)
eval e@(Diag expr) = cached e $ getMatrixDiagonal !(eval expr)
eval e@(Triangle tri expr) = cached e $ triangle !(eval expr) tri
eval e@(Transpose expr) = cached e $ transpose !(eval expr) [1, 0]
eval e@(Identity {dtype} n) = cached e $ let n = cast n in do
  MkCachingBuilder builder _ <- get
  identityMatrix {dtype} builder n n
eval e@(Broadcast {dtype} from to expr) = cached e $
  case elem 0 to && from /= to of
    True => do
      MkCachingBuilder builder _ <- get
      literal <- allocLiteral {dtype} to
      constantLiteral builder literal
    _ =>
      let broadcastDims = map (+ length to `minus` length from) $ range $ length from
       in broadcastInDim !(eval expr) to broadcastDims
eval e@(Map exprParams exprf exprs dims) = cached e $ do
  computation <- buildWithSubBuilder "computation" (map eval exprParams) (eval exprf)
  MkCachingBuilder builder _ <- get
  map builder !(traverse eval exprs) computation dims 
eval e@(Reduce p0 p1 exprf neutral axis expr) = cached e $ do
  computation <- buildWithSubBuilder "computation" [(eval p0), (eval p1)] (eval exprf) 
  reduce !(eval expr) !(eval neutral) computation [axis]
eval e@(Sort p0 p1 exprComp axis isStable exprs) = cached e $ do
  comparator <- buildWithSubBuilder "comparator" [(eval p0), (eval p1)] (eval exprComp)
  sort !(traverse eval exprs) comparator axis isStable 
eval e@(Reverse axes expr) = cached e $ rev !(eval expr) axes
eval e@(Eq l r) = cached e $ eq !(eval l) !(eval r)
eval e@(Ne l r) = cached e $ ne !(eval l) !(eval r)
eval e@(Add l r) = cached e $ add !(eval l) !(eval r)
eval e@(Sub l r) = cached e $ sub !(eval l) !(eval r)
eval e@(Mul l r) = cached e $ mul !(eval l) !(eval r)
eval e@(Div l r) = cached e $ div !(eval l) !(eval r)
eval e@(Pow l r) = cached e $ pow !(eval l) !(eval r)
eval e@(Lt l r) = cached e $ lt !(eval l) !(eval r)
eval e@(Gt l r) = cached e $ gt !(eval l) !(eval r)
eval e@(Le l r) = cached e $ le !(eval l) !(eval r)
eval e@(Ge l r) = cached e $ ge !(eval l) !(eval r)
eval e@(And l r) = cached e $ and !(eval l) !(eval r)
eval e@(Or l r) = cached e $ or !(eval l) !(eval r)
eval e@(Min l r) = cached e $ min !(eval l) !(eval r)
eval e@(Max l r) = cached e $ max !(eval l) !(eval r)
eval e@(Not expr) = cached e $ not !(eval expr)
eval e@(Neg expr) = cached e $ neg !(eval expr)
eval e@(Reciprocal expr) = cached e $ reciprocal !(eval expr)
eval e@(Abs expr) = cached e $ abs !(eval expr)
eval e@(Ceil expr) = cached e $ ceil !(eval expr)
eval e@(Floor expr) = cached e $ floor !(eval expr)
eval e@(Exp expr) = cached e $ exp !(eval expr)
eval e@(Log expr) = cached e $ log !(eval expr)
eval e@(Logistic expr) = cached e $ logistic !(eval expr)
eval e@(Erf expr) = cached e $ erf !(eval expr)
eval e@(Square expr) = cached e $ square !(eval expr)
eval e@(Sqrt expr) = cached e $ sqrt !(eval expr)
eval e@(Sin expr) = cached e $ sin !(eval expr)
eval e@(Cos expr) = cached e $ cos !(eval expr)
eval e@(Tan expr) = cached e $ tan !(eval expr)
eval e@(Asin expr) = cached e $ asin !(eval expr)
eval e@(Acos expr) = cached e $ acos !(eval expr)
eval e@(Atan expr) = cached e $ atan !(eval expr)
eval e@(Sinh expr) = cached e $ sinh !(eval expr)
eval e@(Cosh expr) = cached e $ cosh !(eval expr)
eval e@(Tanh expr) = cached e $ tanh !(eval expr)
eval e@(Asinh expr) = cached e $ asinh !(eval expr)
eval e@(Acosh expr) = cached e $ acosh !(eval expr)
eval e@(Atanh expr) = cached e $ atanh !(eval expr)
eval e@(Select pred true false) = cached e $ select !(eval pred) !(eval true) !(eval false)
eval e@(Cond pred pt exprTrue true pf exprFalse false) = cached e $ do
  trueComp <- buildWithSubBuilder "truthy computation" [eval pt] (eval exprTrue)
  falseComp <- buildWithSubBuilder "falsy computation" [eval pf] (eval exprFalse)
  conditional !(eval pred) !(eval true) trueComp !(eval false) falseComp
eval e@(Dot l r) = cached e $ dot !(eval l) !(eval r)
eval e@(Cholesky expr) = cached e $ cholesky !(eval expr) True
eval e@(TriangularSolve a b leftSide lower unitDiagonal transposeA) =
  cached e $ triangularSolve !(eval a) !(eval b) leftSide lower unitDiagonal transposeA
eval e@(UniformFloatingPointDistributionValue
    key initialState bitGenerator minval maxval shape
  ) = cached e $ do
  let valueStatePair = do
        uniformFloatingPointDistribution
          !(eval key)
          !(eval initialState)
          ThreeFry
          !(eval minval)
          !(eval maxval)
          !(mkShape {dtype=F64} shape)
  -- are we calculating value and state only once per sample?
  ignore $ map snd valueStatePair
  map fst valueStatePair
eval e@(UniformFloatingPointDistributionState
    key initialState bitGenerator minval maxval shape
  ) = cached e $ do
  let valueStatePair = do
        uniformFloatingPointDistribution
          !(eval key)
          !(eval initialState)
          ThreeFry
          !(eval minval)
          !(eval maxval)
          !(mkShape {dtype=F64} shape)
  ignore $ map fst valueStatePair
  map snd valueStatePair
eval e@(NormalFloatingPointDistributionValue key initialState bitGenerator shape) = cached e $ do
  let valueStatePair = do
        normalFloatingPointDistribution
          !(eval key) !(eval initialState) bitGenerator !(mkShape {dtype=F64} shape)
  ignore $ map snd valueStatePair
  map fst valueStatePair
eval e@(NormalFloatingPointDistributionState key initialState bitGenerator shape) = cached e $ do
  let valueStatePair = do
        normalFloatingPointDistribution
          !(eval key) !(eval initialState) bitGenerator !(mkShape {dtype=F64} shape)
  ignore $ map fst valueStatePair
  map snd valueStatePair

export
run : PrimitiveRW dtype a => Expr -> {shape : _} -> Literal shape a
run expr = unsafePerformIO $ do
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  computation <- build "" (assert_total $ eval expr)
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
