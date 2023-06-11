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

export
data Err = IndexErr String

export
Show Err where
  show (IndexErr msg) = "IndexErr: \{msg}"

Computation : Type -> Type
Computation = StateT (SortedMap Nat XlaOp) (EitherT Err IO)

lookup : Nat -> Computation XlaOp
lookup n = do
  case lookup n !get of
       Nothing =>
         lift $ left (IndexErr "Tried to look up value at index \{show n} but none was found.")
       Just op => pure op

interpret : XlaBuilder -> Nat -> Env -> Computation XlaOp

buildSub : XlaBuilder -> String -> Fn arity -> Computation XlaComputation
buildSub builder name (MkFn params i env) = do
  subBuilder <- createSubBuilder builder name
  traverse_ (interpretParameter subBuilder) (enumerate params)
  root <- assert_total $ interpret subBuilder i env
  build subBuilder root

  where

  interpretParameter : XlaBuilder -> (Nat, Nat, ShapeAndType) -> Computation ()
  interpretParameter builder (position, i, MkShapeAndType shape dtype) = do
    xlaShape <- mkShape {dtype} shape
    param <- parameter builder position xlaShape name
    put $ insert i param !get

covering
enqueue : XlaBuilder -> Expr -> Computation XlaOp
enqueue builder (FromLiteral {dtype} lit) = constantLiteral builder !(write {dtype} lit)
enqueue _       (Arg x) = lookup x
enqueue builder (Tuple xs) = tuple builder !(traverse lookup xs)
enqueue builder (GetTupleElement idx x) = getTupleElement !(lookup x) idx
enqueue builder (MinValue {dtype}) = minValue {dtype} builder
enqueue builder (MaxValue {dtype}) = maxValue {dtype} builder
enqueue builder (MinFiniteValue {dtype}) = minFiniteValue {dtype} builder
enqueue builder (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} builder
enqueue _       (ConvertElementType x) = convertElementType {dtype = F64} !(lookup x)
enqueue _       (Reshape from to x) = reshape !(lookup x) (range $ length from) to
enqueue _       (Slice starts stops strides x) = slice !(lookup x) starts stops strides
enqueue _       (DynamicSlice starts sizes x) =
  dynamicSlice !(lookup x) !(traverse lookup starts) sizes
enqueue builder (Concat axis x y) = concatInDim builder [!(lookup x), !(lookup y)] (cast axis)
enqueue _       (Diag x) = getMatrixDiagonal !(lookup x)
enqueue _       (Triangle tri x) = triangle !(lookup x) tri
enqueue _       (Transpose ordering x) = transpose !(lookup x) ordering
enqueue builder (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} builder n n
enqueue builder (Broadcast {dtype} from to x) =
  if elem 0 to && from /= to
  then do
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
   let broadcastDims = map (+ length to `minus` length from) $ range $ length from
    in broadcastInDim !(lookup x) to broadcastDims
enqueue builder (Map f xs dims) = do
  computation <- buildSub builder "computation" f
  map builder (toList !(traverse lookup xs)) computation dims
enqueue builder (Reduce f neutral axes x) = do
  computation <- buildSub builder "computation" f
  reduce !(lookup x) !(lookup neutral) computation axes
enqueue builder (Sort f axis isStable xs) = do
  comparator <- buildSub builder "comparator" f
  sort !(traverse lookup xs) comparator axis isStable 
enqueue _       (Reverse axes x) = rev !(lookup x) axes
enqueue _       (BinaryElementwise f x y) = toXla f !(lookup x) !(lookup y)
  where
  toXla : BinaryOp -> HasIO io => XlaOp -> XlaOp -> io XlaOp
  toXla = \case
    Eq  => eq
    Ne  => ne
    Add => add
    Sub => sub
    Mul => mul
    Div => div
    Rem => rem
    Pow => pow
    Lt  => lt
    Gt  => gt
    Le  => le
    Ge  => ge
    And => and
    Or  => or
    Min => min
    Max => max
enqueue _       (UnaryElementwise f x) = toXla f !(lookup x)
  where
  toXla : UnaryOp -> HasIO io => XlaOp -> io XlaOp
  toXla = \case
    Not        => not
    Neg        => neg
    Reciprocal => reciprocal
    Ceil       => ceil
    Floor      => floor
    Abs        => abs
    Log        => log
    Exp        => exp
    Logistic   => logistic
    Erf        => erf
    Square     => square
    Sqrt       => sqrt
    Sin        => sin
    Cos        => cos
    Tan        => tan
    Asin       => asin
    Acos       => acos
    Atan       => atan
    Sinh       => sinh
    Cosh       => cosh
    Tanh       => tanh
    Asinh      => asinh
    Acosh      => acosh
    Atanh      => atanh
enqueue _       (Argmin {out} axis x) = argMin {outputType=out} !(lookup x) axis
enqueue _       (Argmax {out} axis x) = argMax {outputType=out} !(lookup x) axis
enqueue _       (Select pred true false) = select !(lookup pred) !(lookup true) !(lookup false)
enqueue builder (Cond pred fTrue true fFalse false) = do
  trueComp <- buildSub builder "truthy computation" fTrue
  falseComp <- buildSub builder "falsy computation" fFalse
  conditional !(lookup pred) !(lookup true) trueComp !(lookup false) falseComp
enqueue _       (Dot l r) = dot !(lookup l) !(lookup r)
enqueue _       (Cholesky x) = cholesky !(lookup x) True
enqueue _       (TriangularSolve a b lower) =
  triangularSolve !(lookup a) !(lookup b) True lower False NoTranspose
enqueue builder (UniformFloatingPoint key initialState minval maxval shape) = do
  rngOutput <- uniformFloatingPointDistribution
    !(lookup key)
    !(lookup initialState)
    ThreeFry
    !(lookup minval)
    !(lookup maxval)
    !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]
enqueue builder (UniformUInt key initialState minval maxval shape) = do
  rngOutput <- uniformIntDistribution
    !(lookup key)
    !(lookup initialState)
    ThreeFry
    !(lookup minval)
    !(lookup maxval)
    !(mkShape {dtype=U64} shape)
  tuple builder [value rngOutput, state rngOutput]
enqueue builder (NormalFloatingPoint key initialState shape) = do
  rngOutput <- normalFloatingPointDistribution
    !(lookup key) !(lookup initialState) ThreeFry !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]

interpret builder root env = do
  traverse_ interpretExpr (toList env)
  lookup root

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
