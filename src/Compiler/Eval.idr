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

%hide Util.List.All.map

export
data Err = IndexErr String

export
Show Err where
  show (IndexErr msg) = "IndexErr: \{msg}"

0 Computation : Type -> Type
Computation = StateT (SortedMap Nat XlaComputation, SortedMap Nat XlaOp) (EitherT Err IO)

||| Look up the `XlaOp` at `position` in the graph.
lookup : (position : Nat) -> Computation XlaOp
lookup n = do
  (_, cache) <- get
  case lookup n cache of
       Nothing =>
         lift $ left (IndexErr "Tried to look up XlaOp at index \{show n} but found keys \{show $ toList (keys cache)}")
       Just op => pure op

namespace XlaComputation
  ||| Look up the `XlaComputation` at `position` in the graph.
  lookup : (position : Nat) -> Computation XlaComputation
  lookup n = do
    (cache, _) <- get
    case lookup n cache of
         Nothing =>
           lift $ left (IndexErr "Tried to look up XlaComputation at index \{show n} but found keys \{show $ toList (keys cache)}")
         Just comp => pure comp

interpret : XlaBuilder -> Nat -> Env -> Computation XlaOp

||| Build a computation from an inner function
|||
||| @xlaBuilder The enclosing XLA builder in which this function is built.
|||   This is not the XLA builder used to build the computation itself.
||| @computationName The name of the computation.
||| @arity The function arity.
||| @f The function to build.
buildSub : (xlaBuilder : XlaBuilder) ->
           (computationName : String) ->
           (f : Fn arity) ->
           Computation XlaComputation
buildSub builder name (MkFn params result env) = do
  subBuilder <- createSubBuilder builder name
  traverse_ (interpretParameter subBuilder) (enumerate params)
  root <- assert_total $ interpret subBuilder result env
  build subBuilder root

  where

  interpretParameter : XlaBuilder -> (Nat, Nat, ShapeAndType) -> Computation ()
  interpretParameter builder (positionInFnParams, positionInGraph, MkShapeAndType shape dtype) = do
    xlaShape <- mkShape {dtype} shape
    param <- parameter builder positionInFnParams xlaShape name
    (comps, ops) <- get
    put (comps, insert positionInGraph param ops)

covering
enqueue : XlaBuilder -> Env -> Expr -> Computation XlaOp
enqueue builder _   (FromLiteral {dtype} lit) = constantLiteral builder !(write {dtype} lit)
enqueue _       _   (Arg x) = lookup x
enqueue builder _   (Tuple xs) = tuple builder !(traverse lookup xs)
enqueue builder _   (GetTupleElement idx x) = getTupleElement !(lookup x) idx
enqueue builder env (Call f xs) = do
  (cachedComps, _) <- get
  builtComp <- case lookup f cachedComps of
    Just comp => pure comp
    -- we don't need to index here, we can just build the next function in the list ... it will
    -- be the right one ... but we'll need to know which is the "next"

    -- this works for unnested scenarios, but fails to find the child env when we have deeper
    -- nesting. The env is presumably being stored in the wrong place, or we're trying to
    -- access the nesting in the wrong order
    Nothing => case findChild env f of
      Nothing => lift $ left (IndexErr "Tried to look up child env at index \{show f} with keys \{show $ childKeys env}")
      Just (_ ** comp) => do
        comp <- buildSub builder "name" comp
        (comps, ops) <- get
        put (insert f comp comps, ops)
        pure comp
  call builder builtComp !(traverse lookup xs)
enqueue builder _   (MinValue {dtype}) = minValue {dtype} builder
enqueue builder _   (MaxValue {dtype}) = maxValue {dtype} builder
enqueue builder _   (MinFiniteValue {dtype}) = minFiniteValue {dtype} builder
enqueue builder _   (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} builder
enqueue _       _   (ConvertElementType x) = convertElementType {dtype = F64} !(lookup x)
enqueue _       _   (Reshape from to x) = reshape !(lookup x) (range $ length from) to
enqueue _       _   (Slice starts stops strides x) = slice !(lookup x) starts stops strides
enqueue _       _   (DynamicSlice starts sizes x) =
  dynamicSlice !(lookup x) !(traverse lookup starts) sizes
enqueue builder _   (Concat axis x y) = concatInDim builder [!(lookup x), !(lookup y)] (cast axis)
enqueue _       _   (Diag x) = getMatrixDiagonal !(lookup x)
enqueue _       _   (Triangle tri x) = triangle !(lookup x) tri
enqueue _       _   (Transpose ordering x) = transpose !(lookup x) ordering
enqueue builder _   (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} builder n n
enqueue builder _   (Broadcast {dtype} from to x) =
  if elem 0 to && from /= to
  then do
    literal <- allocLiteral {dtype} to
    constantLiteral builder literal
  else
   let broadcastDims = map (+ length to `minus` length from) $ range $ length from
    in broadcastInDim !(lookup x) to broadcastDims
enqueue builder _   (Map f xs dims) = do
  computation <- buildSub builder "computation" f
  map builder (toList !(traverse lookup xs)) computation dims
enqueue builder _   (Reduce f neutral axes x) = do
  computation <- buildSub builder "computation" f
  reduce !(lookup x) !(lookup neutral) computation axes
enqueue builder _   (Sort f axis isStable xs) = do
  comparator <- buildSub builder "comparator" f
  sort !(traverse lookup xs) comparator axis isStable 
enqueue _       _   (Reverse axes x) = rev !(lookup x) axes
enqueue _       _   (BinaryElementwise f x y) = toXla f !(lookup x) !(lookup y)
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
enqueue _       _   (UnaryElementwise f x) = toXla f !(lookup x)
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
enqueue _       _   (Argmin {out} axis x) = argMin {outputType=out} !(lookup x) axis
enqueue _       _   (Argmax {out} axis x) = argMax {outputType=out} !(lookup x) axis
enqueue _       _   (Select pred true false) = select !(lookup pred) !(lookup true) !(lookup false)
enqueue builder _   (Cond pred fTrue true fFalse false) = do
  trueComp <- buildSub builder "truthy computation" fTrue
  falseComp <- buildSub builder "falsy computation" fFalse
  conditional !(lookup pred) !(lookup true) trueComp !(lookup false) falseComp
enqueue _       _   (Dot l r) = dot !(lookup l) !(lookup r)
enqueue _       _   (Cholesky x) = cholesky !(lookup x) True
enqueue _       _   (TriangularSolve a b lower) =
  triangularSolve !(lookup a) !(lookup b) True lower False NoTranspose
enqueue builder _   (UniformFloatingPoint key initialState minval maxval shape) = do
  rngOutput <- uniformFloatingPointDistribution
    !(lookup key)
    !(lookup initialState)
    ThreeFry
    !(lookup minval)
    !(lookup maxval)
    !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]
enqueue builder _   (NormalFloatingPoint key initialState shape) = do
  rngOutput <- normalFloatingPointDistribution
    !(lookup key) !(lookup initialState) ThreeFry !(mkShape {dtype=F64} shape)
  tuple builder [value rngOutput, state rngOutput]

interpret builder root env = do
  traverse_ interpretExpr (toList env)
  lookup root

  where
  interpretExpr : (Nat, Expr) -> Computation ()
  interpretExpr (n, expr) = do
    (comps, ops) <- get
    put (comps, insert n !(enqueue builder env expr) ops)

export
toString : Nat -> Env -> EitherT Err IO String
toString root env = do
  builder <- mkXlaBuilder "toString"
  xlaOp <- evalStateT (empty, empty) (interpret builder root env)
  pure $ opToString builder xlaOp

export
run : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> EitherT Err IO (Literal shape a)
run root env = do
  builder <- mkXlaBuilder "root" 
  root <- evalStateT (empty, empty) (interpret builder root env)
  computation <- XlaBuilder.build builder root
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
