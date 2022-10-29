{--
Copyright 2022 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either refess or implied.
See the License for the specific language governing permissions and
limitations under the License.
--}
module Compiler.Eval

import Control.Monad.Error.Either
import Control.Monad.Reader
import Data.List
import Data.List.Elem
import Data.SortedMap
import Decidable.Equality

import Compiler.Graph
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

public export 0
Compiled : Type -> Type
Compiled a = EitherT CompilerError IO a

addParams : XlaBuilder -> List FullShape -> Compiled (List XlaOp)
addParams builder xs = traverse (uncurry addParam) (enumerate xs) where
  addParam : Nat -> FullShape -> Compiled XlaOp
  addParam position (MkFullShape shape {dtype}) = do
  xlaShape <- mkShape shape {dtype}
  parameter builder position xlaShape ""

record CompiledOps where
  constructor MkXlaGraph
  params : List XlaOp
  nodes : List XlaOp

enqueue : XlaBuilder -> List CompiledOps -> Node -> Compiled XlaOp

eval : XlaBuilder -> Graph -> List CompiledOps -> Compiled (List XlaOp)
eval builder (MkGraph params nodes) parents = go !(addParams builder params) nodes [] 

  where
 
  go : List XlaOp -> List Node -> List XlaOp -> Compiled (List XlaOp)
  go params [] ops = pure ops
  go params (node :: nodes) ops = do
    op <- enqueue builder (MkXlaGraph params ops :: parents) node
    go params nodes (op :: ops) 

index : Ref -> List CompiledOps -> Compiled XlaOp
index (P scope pos) ops =
  case index scope ops of
       Left err => throwE err
       Right ops => case index pos ops.params of
                         Left err => throwE err
                         Right op => pure op
index (N scope pos) ops = 
  case index scope ops of
       Left err => throwE err
       Right ops => case index pos ops.nodes of
                         Left err => throwE err
                         Right op => pure op
{-
index (P 0 pos) (MkXlaGraph _ params _) =
  case index pos params of
       Left err => throwE err
       Right op => pure op
index (P (S k) pos) (MkXlaGraph [] _ _) = throwE ?index_param_no_parents 
index (P (S k) pos) (MkXlaGraph (p :: _) _ _) = index (P k pos) p
index (N 0 i) (MkXlaGraph _ _ nodes) =
  case index i nodes of
       Left err => throwE err
       Right op => pure op
index (N (S k) i) (MkXlaGraph [] _ _) = throwE ?index_node_no_parents
index (N (S k) i) (MkXlaGraph (p :: _) _ _) = index (N k i) p
-}

enqueue builder _ (FromLiteral {dtype} lit) = do
  literal <- write {dtype} lit
  constantLiteral builder literal
enqueue builder ops (Tuple refs) = tuple builder !(traverse (flip index ops) refs)
enqueue builder ops (GetTupleElement idx ref) = getTupleElement !(index ref ops) idx
enqueue builder _ (MinValue {dtype}) = minValue {dtype} builder
enqueue builder _ (MaxValue {dtype}) = maxValue {dtype} builder
enqueue builder _ (MinFiniteValue {dtype}) = minFiniteValue {dtype} builder
enqueue builder _ (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} builder
enqueue builder ops (ConvertElementType ref) = convertElementType {dtype=F64} !(index ref ops)
enqueue builder ops (Reshape from to ref) = reshape !(index ref ops) (range $ length from) to
enqueue builder ops (Slice starts stops strides ref) = slice !(index ref ops) starts stops strides
enqueue builder ops (DynamicSlice starts sizes ref) =
  dynamicSlice !(index ref ops) !(traverse (flip index ops) starts) sizes
enqueue builder ops (Concat axis ref ref') = concatInDim builder [!(index ref ops), !(index ref' ops)] (cast axis)
enqueue builder ops (Diag ref) = getMatrixDiagonal !(index ref ops)
enqueue builder ops (Triangle tri ref) = triangle !(index ref ops) tri
enqueue builder ops (Transpose ordering ref) = transpose !(index ref ops) ordering
enqueue builder ops (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} builder n n
enqueue builder ops (Broadcast {dtype} from to ref) =
  if elem 0 to && from /= to
  then do
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
    let broadcastDims = map (+ length to `minus` length from) $ range $ length from
     in broadcastInDim !(index ref ops) to broadcastDims
-- enqueue builder ops (Map (MkFn {arity} refParams reff) refs dims) = do
--   computation <- buildWithSubBuilder "computation" (map enqueue $ toList refParams) (enqueue reff)
--   map builder !(traverse enqueue $ toList refs) computation dims
enqueue builder ops (Reduce semigroup neutral axes ref) = do
  subBuilder <- createSubBuilder builder "computation"
  semigroupOps <- eval subBuilder semigroup ops
  case semigroupOps of
       [] => throwE ?reduce_err
       (root :: _) => do
         computation <- build subBuilder root
         reduce !(index ref ops) !(index neutral ops) computation axes
enqueue builder ops (Sort comparator axis isStable refs) = do
  subBuilder <- createSubBuilder builder "computation"
  comparatorOps <- eval subBuilder comparator ops 
  case comparatorOps of
       [] => throwE ?sort_err
       (root :: _) => do
         computation <- build subBuilder root
         sort !(traverse (flip index ops) refs) computation axis isStable
enqueue builder ops (Reverse axes ref) = rev !(index ref ops) axes
enqueue builder ops (Eq l r) = eq !(index l ops) !(index r ops)
enqueue builder ops (Ne l r) = ne !(index l ops) !(index r ops)
enqueue builder ops (Add l r) = add !(index l ops) !(index r ops)
enqueue builder ops (Sub l r) = sub !(index l ops) !(index r ops)
enqueue builder ops (Mul l r) = mul !(index l ops) !(index r ops)
enqueue builder ops (Div l r) = div !(index l ops) !(index r ops)
enqueue builder ops (Pow l r) = pow !(index l ops) !(index r ops)
enqueue builder ops (Lt l r) = lt !(index l ops) !(index r ops)
enqueue builder ops (Gt l r) = gt !(index l ops) !(index r ops)
enqueue builder ops (Le l r) = le !(index l ops) !(index r ops)
enqueue builder ops (Ge l r) = ge !(index l ops) !(index r ops)
enqueue builder ops (And l r) = and !(index l ops) !(index r ops)
enqueue builder ops (Or l r) = or !(index l ops) !(index r ops)
enqueue builder ops (Min l r) = min !(index l ops) !(index r ops)
enqueue builder ops (Max l r) = max !(index l ops) !(index r ops)
enqueue builder ops (Not ref) = not !(index ref ops)
enqueue builder ops (Neg ref) = neg !(index ref ops)
enqueue builder ops (Reciprocal ref) = reciprocal !(index ref ops)
enqueue builder ops (Abs ref) = abs !(index ref ops)
enqueue builder ops (Ceil ref) = ceil !(index ref ops)
enqueue builder ops (Floor ref) = floor !(index ref ops)
enqueue builder ops (Exp ref) = exp !(index ref ops)
enqueue builder ops (Log ref) = log !(index ref ops)
enqueue builder ops (Logistic ref) = logistic !(index ref ops)
enqueue builder ops (Erf ref) = erf !(index ref ops)
enqueue builder ops (Square ref) = square !(index ref ops)
enqueue builder ops (Sqrt ref) = sqrt !(index ref ops)
enqueue builder ops (Sin ref) = sin !(index ref ops)
enqueue builder ops (Cos ref) = cos !(index ref ops)
enqueue builder ops (Tan ref) = tan !(index ref ops)
enqueue builder ops (Asin ref) = asin !(index ref ops)
enqueue builder ops (Acos ref) = acos !(index ref ops)
enqueue builder ops (Atan ref) = atan !(index ref ops)
enqueue builder ops (Sinh ref) = sinh !(index ref ops)
enqueue builder ops (Cosh ref) = cosh !(index ref ops)
enqueue builder ops (Tanh ref) = tanh !(index ref ops)
enqueue builder ops (Asinh ref) = asinh !(index ref ops)
enqueue builder ops (Acosh ref) = acosh !(index ref ops)
enqueue builder ops (Atanh ref) = atanh !(index ref ops)
enqueue builder ops (Argmin {out} axis ref) = argMin {outputType=out} !(index ref ops) axis
enqueue builder ops (Argmax {out} axis ref) = argMax {outputType=out} !(index ref ops) axis
enqueue builder ops (Select pred true false) =
  select !(index pred ops) !(index true ops) !(index false ops)
enqueue builder ops (Cond pred fTrue true fFalse false) = do
  subBuilderTrue <- createSubBuilder builder "truthy computation"
  subBuilderFalse <- createSubBuilder builder "falsy computation"
  fTrueOps <- eval subBuilderTrue fTrue ops
  fFalseOps <- eval subBuilderFalse fFalse ops
  case (fTrueOps, fFalseOps) of
       ([], _) => throwE ?cond_no_true_err
       (_, []) => throwE ?cond_no_false_err
       (rootTrue :: _, rootFalse :: _) => do
         computationTrue <- build subBuilderTrue rootTrue
         computationFalse <- build subBuilderFalse rootFalse
         conditional !(index pred ops) !(index true ops) computationTrue !(index false ops) computationFalse
enqueue builder ops (Dot l r) = dot !(index l ops) !(index r ops)
enqueue builder ops (Cholesky ref) = cholesky !(index ref ops) True
enqueue builder ops (TriangularSolve a b lower) =
  triangularSolve !(index a ops) !(index b ops) True lower False NoTranspose
enqueue builder ops (UniformFloatingPoint key initialState minval maxval shape) = do
  rngOutput <- uniformFloatingPointDistribution
    !(index key ops)
    !(index initialState ops)
    ThreeFry
    !(index minval ops)
    !(index maxval ops)
    !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]
enqueue builder ops (NormalFloatingPoint key initialState shape) = do
  rngOutput <- normalFloatingPointDistribution
    !(index key ops) !(index initialState ops) ThreeFry !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]

-- impl nodes [] where
--  impl : List Node -> List XlaOp -> Compiled (List XlaOp)
--  impl _ ops = pure ops
--  impl (node :: nodes) ops = do
--    op <- enqueue builder ops node
--    impl nodes (op :: ops)

-- impl [[3], [4, 5], Concat 0 1] []
--   t = [3]
--   ts = [[4, 5], Concat 0 1]
--   ops = []
--   op = enqueue [] [3] = ConstantLiteral [3]
--   return impl [[4, 5], Concat 0 1] [ConstantLiteral [3]]
--     t = [4, 5]
--     ts = [Concat 0 1]
--     ops = [ConstantLiteral [3]]
--     op = enqueue [ConstantLiteral [3]] [4, 5] = ConstantLiteral [4, 5]
--     return impl [Concat 0 1] [ConstantLiteral [3], ConstantLiteral [4, 5]]
--       t = Concat 0 1
--       ts = []
--       ops = [ConstantLiteral [3], ConstantLiteral [4, 5]]
--       op = enqueue [ConstantLiteral [3], ConstantLiteral [4, 5]] (Concat 0 1) = ConcatInDim [3] [4, 5]
--       return impl [] [ConstantLiteral [3], ConstantLiteral [4, 5], ConcatInDim [3] [4, 5]]

export
toString : Graph -> Compiled String
toString terms = do
  builder <- mkXlaBuilder "toString"
  ops <- eval builder terms []
  case ops of
    [] => throwE ?noops
    (op :: _) => pure $ opToString builder op

export
run : PrimitiveRW dtype a => Graph -> {shape : _} -> Compiled (Literal shape a)
run graph = do
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  builder <- mkXlaBuilder ""
  ops <- eval builder graph []
  case ops of
       [] => throwE ?run_err
       (root :: _) => do
         computation <- build builder root
         client <- getOrCreateLocalClient platform
         lit <- executeAndTransfer client computation
         pure (read {dtype} lit)
