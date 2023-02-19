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

public export
data Err = IndexErr String

Computation : Type -> Type
Computation = StateT (SortedMap Nat XlaOp) (EitherT Err IO)

lookup : Nat -> SortedMap Nat a -> Computation a
lookup n xs =
  case lookup n xs of
       Nothing =>
         lift $ left (IndexErr "Tried to look up value at index \{show n} but none was found.")
       Just x => pure x

parameter : XlaBuilder -> Nat -> ShapeAndType -> String -> Computation XlaOp
parameter builder position (MkShapeAndType shape dtype) name = do
  xlaShape <- mkShape {dtype} shape
  XlaBuilder.parameter builder position xlaShape name

interpret : XlaBuilder -> Nat -> Env -> Computation XlaOp

covering
enqueue : XlaBuilder -> Expr -> Computation XlaOp
enqueue builder (FromLiteral {dtype} lit) = do
  literal <- write {dtype} lit
  constantLiteral builder literal
enqueue _ (Arg x) = lookup x !get 
enqueue builder (Tuple xs) = do
  tupleOps <- traverse (flip lookup !get) xs
  tuple builder tupleOps
enqueue builder (GetTupleElement idx x) = getTupleElement !(lookup x !get) idx
enqueue builder (MinValue {dtype}) = minValue {dtype} builder
enqueue builder (MaxValue {dtype}) = maxValue {dtype} builder
enqueue builder (MinFiniteValue {dtype}) = minFiniteValue {dtype} builder
enqueue builder (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} builder
-- todo test dtypes here, is this a bug?
enqueue _ (ConvertElementType x) = convertElementType {dtype = F64} !(lookup x !get)
enqueue _ (Reshape from to x) = reshape !(lookup x !get) (range $ length from) to
enqueue _ (Slice starts stops strides x) = slice !(lookup x !get) starts stops strides
enqueue _ (DynamicSlice starts sizes x) =
  dynamicSlice !(lookup x !get) !(traverse (flip lookup !get) starts) sizes
enqueue builder (Concat axis x y) =
  concatInDim builder [!(lookup x !get), !(lookup y !get)] (cast axis)
enqueue _ (Diag x) = getMatrixDiagonal !(lookup x !get)
enqueue _ (Triangle tri x) = triangle !(lookup x !get) tri
enqueue _ (Transpose ordering x) = transpose !(lookup x !get) ordering
enqueue builder (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} builder n n
enqueue builder (Broadcast {dtype} from to x) =
  if elem 0 to && from /= to
  then do
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
   let broadcastDims = map (+ length to `minus` length from) $ range $ length from
    in broadcastInDim !(lookup x !get) to broadcastDims
{-
enqueue builder ops (Map (MkFn {arity} shapesAndTypes result) xs dims) = do
  computation <- buildWithSubBuilder "computation" (map enqueue $ toList exprParams) (enqueue exprf)
  (builder, _) <- get
  map builder !(traverse enqueue $ toList exprs) computation dims 
  -}
enqueue builder (Reduce (MkFn [(i0, p0), (i1, p1)] j env) neutral axes x) = do
  subBuilder <- createSubBuilder builder "computation"
  put $ insert i0 !(parameter subBuilder 0 p0 "") !get
  put $ insert i1 !(parameter subBuilder 1 p1 "") !get
  root <- assert_total $ interpret subBuilder j env
  computation <- XlaBuilder.build subBuilder root
  reduce !(lookup x !get) !(lookup neutral !get) computation axes
enqueue builder (Sort (MkFn [(i0, p0), (i1, p1)] j env) axis isStable xs) = do
  subBuilder <- createSubBuilder builder "comparator"
  put $ insert i0 !(parameter subBuilder 0 p0 "") !get
  put $ insert i1 !(parameter subBuilder 1 p1 "") !get
  root <- assert_total $ interpret subBuilder j env
  comparator <- XlaBuilder.build subBuilder root
  sort !(traverse (flip lookup !get) xs) comparator axis isStable 
enqueue _ (Reverse axes x) = rev !(lookup x !get) axes
enqueue _ (BinaryElementwise f x y) = toXla f !(lookup x !get) !(lookup y !get)
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
enqueue _ (UnaryElementwise f x) = toXla f !(lookup x !get)
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
enqueue _ (Argmin {out} axis x) = argMin {outputType=out} !(lookup x !get) axis
enqueue _ (Argmax {out} axis x) = argMax {outputType=out} !(lookup x !get) axis
enqueue _ (Select pred true false) =
  select !(lookup pred !get) !(lookup true !get) !(lookup false !get)
{-
enqueue builder ops (Cond pred (MkFn [pt] exprTrue) true (MkFn [pf] exprFalse) false) = ?cond -- do
  trueComp <- buildWithSubBuilder "truthy computation" [enqueue pt] (enqueue exprTrue)
  falseComp <- buildWithSubBuilder "falsy computation" [enqueue pf] (enqueue exprFalse)
  conditional !(enqueue pred) !(enqueue true) trueComp !(enqueue false) falseComp
-}
enqueue _ (Dot l r) = dot !(lookup l !get) !(lookup r !get)
enqueue _ (Cholesky x) = cholesky !(lookup x !get) True
enqueue _ (TriangularSolve a b lower) =
  triangularSolve !(lookup a !get) !(lookup b !get) True lower False NoTranspose
enqueue builder (UniformFloatingPoint key initialState minval maxval shape) = ?ufp -- do
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
enqueue builder (NormalFloatingPoint key initialState shape) = ?nfp -- do
  {-
  rngOutput <- normalFloatingPointDistribution
    !(enqueue key) !(enqueue initialState) ThreeFry !(mkShape {dtype=F64} shape)
  (builder, _) <- get
  tuple builder [value rngOutput, state rngOutput]
  -}

interpret builder root env = do
  traverse_ interpretExpr (toList env)
  lookup root !get 

  where
  interpretExpr : (Nat, Expr) -> Computation ()
  interpretExpr (n, expr) = put (insert n !(enqueue builder expr) !get)

export
toString : Nat -> Env -> EitherT Err IO String
toString root env = do
  builder <- mkXlaBuilder "toString"
  xlaOp <- evalStateT empty (interpret builder root env)
  pure $ opToString builder xlaOp

export
run : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> EitherT Err IO (Literal shape a)
run root env = do
  builder <- mkXlaBuilder "root" 
  root <- evalStateT empty (interpret builder root env)
  computation <- XlaBuilder.build builder root
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)

