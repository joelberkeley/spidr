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
import Control.Monad.Trans
import Data.IOArray
import Data.List
import Data.List.Elem

import Compiler.Enzyme.MLIR.Dialect.Dialect
import Compiler.Enzyme.MLIR.Implementations.CoreDialectsAutoDiffImplementations
import Compiler.Enzyme.MLIR.Passes.Passes
import Compiler.EnzymeJAX.Src.EnzymeAD.JAX.Implementations.XLADerivatives
import Compiler.EnzymeJAX.Src.EnzymeAD.JAX.Passes.Passes
import Compiler.LLVM.Support.RawOStream
import Compiler.MLIR.Dialect.Func.IR.FuncOps
import Compiler.MLIR.IR
import Compiler.MLIR.Pass.PassManager
import Compiler.MLIR.Pass.PassRegistry
import Compiler.MLIR.Transforms.Passes
import Compiler.Stablehlo.Dialect.Register
import Compiler.Stablehlo.Dialect.StablehloOps
import Compiler.Xla.Client.ExecutableBuildOptions
import Compiler.Xla.HLO.Builder.XlaBuilder
import Compiler.Xla.HLO.Builder.XlaComputation
import Compiler.Xla.HLO.Builder.Lib.Arithmetic
import Compiler.Xla.HLO.Builder.Lib.Constants
import Compiler.Xla.HLO.Builder.Lib.Math
import Compiler.Xla.HLO.Builder.Lib.Matrix
import Compiler.Xla.HLO.Builder.Lib.PRNG
import Compiler.Xla.HLO.Translate.StableHLO
import Compiler.Xla.MLIRHLO.MHLO.IR.Register
import Compiler.Xla.PJRT.C.PjrtCApi
import Compiler.Xla.PJRT.PjrtExecutable
import Compiler.Xla.Service.HloProto
import Compiler.Xla.Literal
import Compiler.Xla.Shape
import Compiler.Xla.ShapeUtil
import Compiler.Xla.XlaData
import Compiler.Expr
import Compiler.FFI
import Compiler.LiteralRW
import Literal
import Primitive
import Types
import Util
import Device

export
data Err
  = OutOfBounds Nat Nat
  | ValueNotFound Nat
  | PjrtErr PjrtError
  | MlirPassError String

export
Show Err where
  show (OutOfBounds idx size) = "Index \{show idx} is out of bounds for array of size \{show size}"
  show (ValueNotFound idx) = "Value not found at index \{show idx}"
  show (PjrtErr err) = show err
  show (MlirPassError err) = "MlirPassError: \{err}"

public export 0
ErrIO : Type -> Type
ErrIO = EitherT Err IO

hloModuleProtoToStableHLO : MLIRContext -> HloModuleProto -> ErrIO ModuleOp
hloModuleProtoToStableHLO mlirCtx proto = do
  dialectRegistry <- mkDialectRegistry
  registerAllMhloDialects dialectRegistry
  registerAllDialects dialectRegistry
  appendDialectRegistry mlirCtx dialectRegistry
  convertHloToStablehlo mlirCtx proto

covering
interpret : IOArray XlaOp => XlaBuilder -> Fn arity -> ErrIO XlaOp

covering
compile : IOArray XlaOp => XlaBuilder -> Fn arity -> ErrIO XlaComputation
compile xlaBuilder f = build xlaBuilder =<< interpret xlaBuilder f

interpret @{cache} xlaBuilder (MkFn params root env) = do
  traverse_ interpretParameter (enumerate params)
  traverse_ (\(i, expr) => do set i !(interpretE expr)) (toList env)
  interpretE root

  where

  set : Nat -> XlaOp -> ErrIO ()
  set idx xlaOp = do
    False <- writeArray cache (cast idx) xlaOp | True => right ()
    left $ OutOfBounds idx (cast $ max cache)

  get : Nat -> ErrIO XlaOp
  get idx = do
    Nothing <- readArray cache (cast idx) | Just op => right op
    left $ let max = cast (max cache) in if idx >= max then OutOfBounds idx max else ValueNotFound idx

  interpretParameter : (Nat, Nat, Parameter) -> ErrIO ()
  interpretParameter (fPos, graphPos, MkParameter shape dtype) = do
    xlaShape <- mkShape {dtype} shape
    param <- parameter xlaBuilder fPos xlaShape (show fPos)
    set graphPos param

  interpretE : Expr -> ErrIO XlaOp
  interpretE (FromLiteral {dtype} lit) = constantLiteral xlaBuilder !(write {dtype} [] lit)
  interpretE (Var x) = get x
  interpretE (Tuple xs) = tuple xlaBuilder !(traverse interpretE xs)
  interpretE (GetTupleElement idx x) = getTupleElement !(interpretE x) idx
  interpretE (Grad shape f x) = do
    -- compile to XLA HLO
    subBuilder <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/grad:f"
    computation <- compile subBuilder f

    -- convert to StableHLO
    mlirCtx <- mkMLIRContext
    stablehlo <- hloModuleProtoToStableHLO mlirCtx !(proto computation)

    -- generate derivative
    loadDialectEnzymeDialect mlirCtx
    registerEnzymePasses
    dialectRegistry <- mkDialectRegistry
    registerCoreDialectAutodiffInterfaces dialectRegistry
    registerStableHLODialectAutoDiffInterface dialectRegistry
    registerCHLODialectAutoDiffInterface dialectRegistry
    insertEnzymeDialect dialectRegistry
    appendDialectRegistry mlirCtx dialectRegistry
    passManager <- mkPassManager mlirCtx
    let enzymePass = "enzyme-wrap{infn=main outfn=fdiff argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined}"
    pipelineParseErrMsg <- cppString
    True <- parsePassPipeline enzymePass passManager !(rawStringOStream pipelineParseErrMsg)
      | False => throwE $ MlirPassError """
        Failed to parse enzyme autodiff pass directive
        \{!(toString pipelineParseErrMsg)}
        """
    CppString.delete pipelineParseErrMsg
    addCanonicalizerPass passManager
    addCSEPass passManager
    addRemoveUnusedEnzymeOpsPass passManager
    addArithRaisingPass passManager
    True <- run passManager !(getOperation stablehlo)
      | False => throwE $ MlirPassError "Failed to run autodiff MLIR passes"

    -- replace main function
    erase !(lookupSymbolIn !(getOperation stablehlo) "main")
    tensorShape <- RankedTensorType.get shape (cast !(getF64Type !(mkOpBuilder mlirCtx)))
    tensorShapeL <- mkTypeRange [cast tensorShape]
    funcType <- FunctionType.get mlirCtx tensorShapeL tensorShapeL
    funcOp <- FuncOp.create !(UnknownLoc.get mlirCtx) "main" funcType
    pushBack stablehlo !(getOperation funcOp)
    entryBlock <- addEntryBlock funcOp
    blockBuilder <- atBlockEnd entryBlock
    scalarShape <- RankedTensorType.get [] (cast !(getF64Type blockBuilder))
    scalarOne <- DenseElementsAttr.get (cast scalarShape) 1.0
    revInit <- createConstantOp blockBuilder !(UnknownLoc.get mlirCtx) scalarOne
    diffOperands <- mkValueRange
      [cast !(getArgument entryBlock 0), cast !(getOpResult !(getOperation revInit) 0)]
    fdiffCallOp <- createCallOp
      blockBuilder !(UnknownLoc.get mlirCtx) "fdiff" tensorShapeL diffOperands
    _ <- createReturnOp
      blockBuilder !(UnknownLoc.get mlirCtx) !(getOpResults !(getOperation fdiffCallOp))

    -- convert back to XLA HLO, and call
    hloProto <- convertStablehloToHlo stablehlo
    computation <- mkXlaComputation hloProto
    call xlaBuilder computation [!(interpretE x)]
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
    subBuilder <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/map:op"
    computation <- compile subBuilder f
    map xlaBuilder (toList !(traverse interpretE xs)) computation dims
  interpretE (Reduce f neutral axes x) = do
    subBuilder <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/reduce:semigroup"
    computation <- compile subBuilder f
    reduce !(interpretE x) !(interpretE neutral) computation axes
  interpretE (Sort f axis isStable xs) = do
    subBuilder <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/sort:compare"
    computation <- compile subBuilder f
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
  interpretE (Argmax {out} axis x) = argMax {outputType = out} !(interpretE x) axis
  interpretE (Select pred true false) =
    select !(interpretE pred) !(interpretE true) !(interpretE false)
  interpretE (Cond pred fTrue true fFalse false) = do
    subBuilderT <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/cond:true"
    subBuilderF <- createSubBuilder xlaBuilder "\{!(name xlaBuilder)}/cond:false"
    compTrue <- compile subBuilderT fTrue
    compFalse <- compile subBuilderF fFalse
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
      !(mkShape {dtype = F64} shape)
    tuple xlaBuilder [value rngOutput, state rngOutput]
  interpretE (NormalFloatingPoint key initialState shape) = do
    rngOutput <- normalFloatingPointDistribution
      !(interpretE key) !(interpretE initialState) ThreeFry !(mkShape {dtype = F64} shape)
    tuple xlaBuilder [value rngOutput, state rngOutput]

||| It is up to the caller to free the `Literal`s.
export covering
execute : Device -> Fn 0 -> {outputs : _} -> Vect outputs Xla.Shape -> ErrIO $ Vect outputs Literal
execute (MkDevice api client) f@(MkFn _ _ env) shapes = do
  xlaBuilder <- mkXlaBuilder "root"
  computation <- compile @{!(newArray $ cast $ counter env)} xlaBuilder f
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
