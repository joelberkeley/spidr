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
module Compiler.Eval

import Control.Monad.Either
import Control.Monad.Reader
import Control.Monad.Trans
import Data.IOArray
import Data.List
import Data.List.Elem

import Compiler.Expr
import Compiler.FFI
import Compiler.LiteralRW
import Compiler.Xla.Client.Lib.Arithmetic
import Compiler.Xla.Client.Lib.Constants
import Compiler.Xla.Client.Lib.Math
import Compiler.Xla.Client.Lib.Matrix
import Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.Client.ExecutableBuildOptions
import Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.Client.XlaComputation
import Compiler.Xla.PJRT.C.PjrtCApi
import Compiler.Xla.PJRT.PjrtExecutable
import Compiler.Xla.Literal
import Compiler.Xla.Shape
import Compiler.Xla.ShapeUtil
import Compiler.Xla.XlaData
import Literal
import Primitive
import Types
import Util
import Device

export
data Err = OutOfBounds Nat Nat
         | ValueNotFound Nat
         | PjrtErr PjrtError

export
Show Err where
  show (OutOfBounds idx size) = "Index \{show idx} is out of bounds for array of size \{show size}"
  show (ValueNotFound idx) = "Value not found at index \{show idx}"
  show (PjrtErr err)= show err

public export 0
ErrIO : Type -> Type
ErrIO = EitherT Err IO

covering
interpret : XlaBuilder -> Fn arity -> ErrIO XlaOp

covering
compile : XlaBuilder -> Fn arity -> ErrIO XlaComputation
compile xlaBuilder f = do
  root <- interpret xlaBuilder f
  build xlaBuilder root

interpret xlaBuilder (MkFn params root env) = do
  let (max, exprs) = toList env
  runReaderT !(newArray $ cast $ max + length params) $ do
    traverse_ interpretParameter (enumerate params)
    traverse_ (\(i, expr) => do set (i + length params) !(interpretE expr)) exprs
    interpretE root

  where

  0 Builder : Type -> Type
  Builder = ReaderT (IOArray XlaOp) ErrIO

  set : Nat -> XlaOp -> Builder ()
  set idx xlaOp = do
    cache <- ask
    True <- lift $ writeArray cache (cast idx) xlaOp
      | False => lift $ left $ OutOfBounds idx (cast $ max cache)
    pure ()

  get : Nat -> Builder XlaOp
  get idx = do
    cache <- ask
    Just xlaOp <- lift $ readArray cache (cast idx)
      | _ => lift $ left $ let max = cast (max cache)
                            in if idx >= max then OutOfBounds idx max else ValueNotFound idx
    pure xlaOp

  interpretParameter : (Nat, ShapeAndType) -> Builder ()
  interpretParameter (position, MkShapeAndType shape dtype) = do
    xlaShape <- mkShape {dtype} shape
    param <- parameter xlaBuilder position xlaShape (show position)
    set position param

  interpretE : Expr -> Builder XlaOp
  interpretE (FromLiteral {dtype} lit) = constantLiteral xlaBuilder !(write {dtype} [] lit)
  interpretE (Var x) = get (x + length params)
  interpretE (Arg x) = get x
  interpretE (Tuple xs) = tuple xlaBuilder !(traverse interpretE xs)
  interpretE (GetTupleElement idx x) = getTupleElement !(interpretE x) idx
  interpretE (MinValue {dtype}) = minValue {dtype} xlaBuilder
  interpretE (MaxValue {dtype}) = maxValue {dtype} xlaBuilder
  interpretE (MinFiniteValue {dtype}) = minFiniteValue {dtype} xlaBuilder
  interpretE (MaxFiniteValue {dtype}) = maxFiniteValue {dtype} xlaBuilder
  interpretE (ConvertElementType x) = convertElementType {dtype = F64} !(interpretE x)
  interpretE (Iota {dtype} shape dim) = iota xlaBuilder !(mkShape {dtype} shape) dim
  interpretE (Reshape from to x) = reshape !(interpretE x) (range $ length from) to
  interpretE (Slice starts stops strides x) = slice !(interpretE x) starts stops strides
  interpretE (DynamicSlice starts sizes x) =
    dynamicSlice !(interpretE x) !(traverse interpretE starts) sizes
  interpretE (Concat axis x y) =
    concatInDim xlaBuilder [!(interpretE x), !(interpretE y)] (cast axis)
  interpretE (Diag x) = getMatrixDiagonal !(interpretE x)
  interpretE (Triangle tri x) = triangle !(interpretE x) tri
  interpretE (Transpose ordering x) = transpose !(interpretE x) ordering
  interpretE (Identity {dtype} n) = let n = cast n in identityMatrix {dtype} xlaBuilder n n
  interpretE (Broadcast {dtype} from to x) =
    if elem 0 to && from /= to
    then do
      shape <- mkShape {dtype} to
      literal <- allocLiteral shape
      constantLiteral xlaBuilder literal
    else
     let broadcastDims = Prelude.map (+ length to `minus` length from) $ range $ length from
      in broadcastInDim !(interpretE x) to broadcastDims
  interpretE (Map f xs dims) = do
    subBuilder <- createSubBuilder xlaBuilder "computation"
    computation <- lift $ compile subBuilder f
    map xlaBuilder (toList !(traverse interpretE xs)) computation dims
  interpretE (Reduce f neutral axes x) = do
    subBuilder <- createSubBuilder xlaBuilder "monoid binary op"
    computation <- lift $ compile subBuilder f
    reduce !(interpretE x) !(interpretE neutral) computation axes
  interpretE (Sort f axis isStable xs) = do
    subBuilder <- createSubBuilder xlaBuilder "comparator"
    computation <- lift $ compile subBuilder f
    sort !(traverse interpretE xs) computation axis isStable
  interpretE (Reverse axes x) = rev !(interpretE x) axes
  interpretE (BinaryElementwise f x y) = toXla f !(interpretE x) !(interpretE y)
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
  interpretE (UnaryElementwise f x) = toXla f !(interpretE x)
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
  interpretE (Argmin {out} axis x) = argMin {outputType=out} !(interpretE x) axis
  interpretE (Argmax {out} axis x) = argMax {outputType=out} !(interpretE x) axis
  interpretE (Select pred true false) =
    select !(interpretE pred) !(interpretE true) !(interpretE false)
  interpretE (Cond pred fTrue true fFalse false) = do
    subBuilderT <- createSubBuilder xlaBuilder "truthy computation"
    subBuilderF <- createSubBuilder xlaBuilder "falsy computation"
    compTrue <- lift $ compile subBuilderT fTrue
    compFalse <- lift $ compile subBuilderF fFalse
    conditional !(interpretE pred) !(interpretE true) compTrue !(interpretE false) compFalse
  interpretE (Dot l r) = dot !(interpretE l) !(interpretE r)
  interpretE (DotGeneral lb rb lc rc l r) = do
    dimensionNumbers <- allocDotDimensionNumbers
    traverse_ (addLhsBatchDimensions dimensionNumbers) lb
    traverse_ (addRhsBatchDimensions dimensionNumbers) rb
    traverse_ (addLhsContractingDimensions dimensionNumbers) lc
    traverse_ (addRhsContractingDimensions dimensionNumbers) rc
    dotGeneral !(interpretE l) !(interpretE r) dimensionNumbers
  interpretE (Cholesky x) = cholesky !(interpretE x) True
  interpretE (TriangularSolve a b lower) =
    triangularSolve !(interpretE a) !(interpretE b) True lower False NoTranspose
  interpretE (UniformFloatingPoint key initialState minval maxval shape) = do
    rngOutput <- uniformFloatingPointDistribution
      !(interpretE key)
      !(interpretE initialState)
      ThreeFry
      !(interpretE minval)
      !(interpretE maxval)
      !(mkShape {dtype=F64} shape)
    tuple xlaBuilder [value rngOutput, state rngOutput]
  interpretE (NormalFloatingPoint key initialState shape) = do
    rngOutput <- normalFloatingPointDistribution
      !(interpretE key) !(interpretE initialState) ThreeFry !(mkShape {dtype=F64} shape)
    tuple xlaBuilder [value rngOutput, state rngOutput]

export covering
toString : Fn 0 -> ErrIO String
toString f = do
  xlaBuilder <- mkXlaBuilder "toString"
  root <- interpret xlaBuilder f
  pure $ opToString xlaBuilder root

||| It is up to the caller to free the `Literal`s.
export covering
execute : Device -> Fn 0 -> {outputs : _} -> Vect outputs Xla.Shape -> ErrIO $ Vect outputs Literal
execute (MkDevice api client) f shapes = do
  xlaBuilder <- mkXlaBuilder "root"
  computation <- compile xlaBuilder f
  bimapEitherT PjrtErr id $ do
    code <- serializeAsString computation
    executableBuildOptions <- mkExecutableBuildOptions
    compileOptions <- serializeAsString !(mkCompileOptions executableBuildOptions)
    loadedExec <- pjrtClientCompile api client !(mkPjrtProgram code) compileOptions
    free code
    free compileOptions
    delete executableBuildOptions

    buffers <- pjrtLoadedExecutableExecute api loadedExec outputs
    pjrtLoadedExecutableDestroy api loadedExec

    for (zip buffers shapes) $ \(buffer, shape) => do
      literal <- allocLiteral shape
      -- is this pure?
      -- note we can probably avoid the difficulties around async
      -- by awaiting the event in pjrtBufferToHostBuffer, thus
      -- making that function synchronous
      event <- pjrtBufferToHostBuffer api buffer literal
      pjrtEventAwait api event
      pjrtEventDestroy api event
      pjrtBufferDestroy api buffer

      pure literal
