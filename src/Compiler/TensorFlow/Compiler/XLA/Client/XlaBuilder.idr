{--
Copyright 2021 Joel Berkeley

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
module Compiler.TensorFlow.Compiler.XLA.Client.XlaBuilder

import System.FFI

import Compiler.FFI
import Compiler.Foreign.TensorFlow.Compiler.XLA.Client.XlaBuilder
import Compiler.TensorFlow.Compiler.XLA.Client.XlaComputation
import Compiler.TensorFlow.Compiler.XLA.Literal
import Compiler.TensorFlow.Compiler.XLA.Shape
import Types
import Util

public export
data XlaBuilder : Type where
  MkXlaBuilder : GCAnyPtr -> XlaBuilder

public export
data XlaOp : Type where
  MkXlaOp : GCAnyPtr -> XlaOp

namespace XlaBuilder
  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . XlaBuilder.prim__delete

namespace XlaOp
  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . XlaOp.prim__delete

export
mkXlaBuilder : HasIO io => String -> io XlaBuilder
mkXlaBuilder computationName = do
  ptr <- primIO (prim__mkXlaBuilder computationName)
  ptr <- onCollectAny ptr XlaBuilder.delete
  pure (MkXlaBuilder ptr)

export
createSubBuilder : HasIO io => XlaBuilder -> String -> io XlaBuilder
createSubBuilder (MkXlaBuilder builderPtr) computationName = do
  subBuilderPtr <- primIO (prim__createSubBuilder builderPtr computationName)
  subBuilderPtr <- onCollectAny subBuilderPtr XlaBuilder.delete
  pure (MkXlaBuilder subBuilderPtr)

export
build : HasIO io => XlaBuilder -> io XlaComputation
build (MkXlaBuilder ptr) = do
  let computationPtr = prim__build ptr
  computationPtr <- onCollectAny computationPtr XlaComputation.delete
  pure (MkXlaComputation computationPtr)

export
opToString : XlaBuilder -> XlaOp -> String
opToString (MkXlaBuilder builderPtr) (MkXlaOp opPtr) = prim__opToString builderPtr opPtr

data XlaOpArray : Type where
  MkXlaOpArray : GCAnyPtr -> XlaOpArray

export
mkXlaOpArray : HasIO io => List XlaOp -> io XlaOpArray
mkXlaOpArray ops = do
  arr <- malloc (cast (length ops) * sizeOfXlaOp)
  traverse_ (\(idx, (MkXlaOp opPtr)) =>
    primIO $ prim__setArrayXlaOp arr (cast idx) opPtr) (enumerate (fromList ops))
  arr <- onCollectAny arr free
  pure (MkXlaOpArray arr)

export
parameter : HasIO io => XlaBuilder -> Nat -> XLA.Shape -> String -> io XlaOp
parameter (MkXlaBuilder builderPtr) position (MkShape shapePtr) name = do
  opPtr <- primIO $ prim__parameter builderPtr (cast position) shapePtr name
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
constantLiteral : HasIO io => XlaBuilder -> Literal -> io XlaOp
constantLiteral (MkXlaBuilder builderPtr) (MkLiteral literalPtr) = do
  opPtr <- primIO (prim__constantLiteral builderPtr literalPtr)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
broadcast : HasIO io => XlaOp -> List Nat -> io XlaOp
broadcast (MkXlaOp opPtr) broadcastSizes = do
  MkIntArray broadcastSizesArrayPtr <- mkIntArray broadcastSizes
  opPtr <- primIO $ prim__broadcast opPtr broadcastSizesArrayPtr (cast $ length broadcastSizes)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
broadcastInDim : HasIO io => XlaOp -> List Nat -> List Nat -> io XlaOp
broadcastInDim (MkXlaOp opPtr) outDimSize broadcastDimensions = do
  MkIntArray outDimSizeArrayPtr <- mkIntArray outDimSize
  MkIntArray broadcastDimensionsArrayPtr <- mkIntArray broadcastDimensions
  opPtr <- primIO $ prim__broadcastInDim
    opPtr
    outDimSizeArrayPtr (cast $ length outDimSize)
    broadcastDimensionsArrayPtr (cast $ length broadcastDimensions)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
reshape : HasIO io => XlaOp -> List Nat -> List Nat -> io XlaOp
reshape (MkXlaOp opPtr) dimensions newSizes = do
  MkIntArray dimensionsArrayPtr <- mkIntArray dimensions
  MkIntArray newSizesArrayPtr <- mkIntArray newSizes
  opPtr <- primIO $ prim__reshape
    opPtr
    dimensionsArrayPtr (cast $ length dimensions)
    newSizesArrayPtr (cast $ length newSizes)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
slice : HasIO io => XlaOp -> List Nat -> List Nat -> List Nat -> io XlaOp
slice (MkXlaOp opPtr) startIndices limitIndices strides = do
  MkIntArray startIndicesArrayPtr <- mkIntArray startIndices
  MkIntArray limitIndicesArrayPtr <- mkIntArray limitIndices
  MkIntArray stridesArrayPtr <- mkIntArray strides
  let rank = cast (length startIndices)
  opPtr <- primIO $ prim__slice
    opPtr
    startIndicesArrayPtr rank
    limitIndicesArrayPtr rank
    stridesArrayPtr rank
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
concatInDim :
  HasIO io =>
  XlaBuilder ->
  (operands : List XlaOp) ->
  {auto 0 _ : NonEmpty operands} ->
  Nat ->
  io XlaOp
concatInDim (MkXlaBuilder builder) operands dimension = do
  MkXlaOpArray xlaOpArrayPtr <- mkXlaOpArray operands
  opPtr <- primIO $ prim__concatInDim
    builder xlaOpArrayPtr (cast $ length operands) (cast dimension)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
select : HasIO io => XlaOp -> XlaOp -> XlaOp -> io XlaOp
select (MkXlaOp pred) (MkXlaOp onTrue) (MkXlaOp onFalse) = do
  opPtr <- primIO $ prim__select pred onTrue onFalse
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

binaryOp : HasIO io => (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr) -> XlaOp -> XlaOp -> io XlaOp
binaryOp prim__f (MkXlaOp x) (MkXlaOp y) = do
  opPtr <- primIO $ prim__f x y
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
eq : HasIO io => XlaOp -> XlaOp -> io XlaOp
eq = binaryOp prim__eq

export
ne : HasIO io => XlaOp -> XlaOp -> io XlaOp
ne = binaryOp prim__ne

export
ge : HasIO io => XlaOp -> XlaOp -> io XlaOp
ge = binaryOp prim__ge

export
gt : HasIO io => XlaOp -> XlaOp -> io XlaOp
gt = binaryOp prim__gt

export
lt : HasIO io => XlaOp -> XlaOp -> io XlaOp
lt = binaryOp prim__lt

export
le : HasIO io => XlaOp -> XlaOp -> io XlaOp
le = binaryOp prim__le

export
dot : HasIO io => XlaOp -> XlaOp -> io XlaOp
dot = binaryOp prim__dot

public export
data Transpose = NoTranspose | Transpose_ | Adjoint

export
triangularSolve : HasIO io => XlaOp -> XlaOp -> Bool -> Bool -> Bool -> Transpose -> io XlaOp
triangularSolve (MkXlaOp a) (MkXlaOp b) leftSide lower unitDiagonal transposeA = do
  let transposeA : Int = case transposeA of
        NoTranspose => 1
        Transpose_ => 2
        Adjoint => 3
  opPtr <- primIO $ prim__triangularSolve
    a b (boolToCInt leftSide) (boolToCInt lower) (boolToCInt unitDiagonal) transposeA
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
cholesky : HasIO io => XlaOp -> Bool -> io XlaOp
cholesky (MkXlaOp a) lower = do
  opPtr <- primIO $ prim__cholesky a (boolToCInt lower)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
add : HasIO io => XlaOp -> XlaOp -> io XlaOp
add = binaryOp prim__add

export
sub : HasIO io => XlaOp -> XlaOp -> io XlaOp
sub = binaryOp prim__sub

export
mul : HasIO io => XlaOp -> XlaOp -> io XlaOp
mul = binaryOp prim__mul

export
div : HasIO io => XlaOp -> XlaOp -> io XlaOp
div = binaryOp prim__div

export
max : HasIO io => XlaOp -> XlaOp -> io XlaOp
max = binaryOp prim__max

export
min : HasIO io => XlaOp -> XlaOp -> io XlaOp
min = binaryOp prim__min

export
and : HasIO io => XlaOp -> XlaOp -> io XlaOp
and = binaryOp prim__and

export
or : HasIO io => XlaOp -> XlaOp -> io XlaOp
or = binaryOp prim__or

export
unaryOp : HasIO io => (GCAnyPtr -> PrimIO AnyPtr) -> XlaOp -> io XlaOp
unaryOp prim__f (MkXlaOp x) = do
  opPtr <- primIO $ prim__f x
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
not : HasIO io => XlaOp -> io XlaOp
not = unaryOp prim__not

export
reduce : HasIO io => XlaOp -> XlaOp -> XlaComputation -> List Nat -> io XlaOp
reduce (MkXlaOp operand) (MkXlaOp initValue) (MkXlaComputation computation) dimensions = do
  MkIntArray dimensionsIntArrayPtr <- mkIntArray dimensions
  opPtr <- primIO $ prim__reduce
    operand initValue computation dimensionsIntArrayPtr (cast $ length dimensions)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
abs : HasIO io => XlaOp -> io XlaOp
abs = unaryOp prim__abs

export
exp : HasIO io => XlaOp -> io XlaOp
exp = unaryOp prim__exp

export
floor : HasIO io => XlaOp -> io XlaOp
floor = unaryOp prim__floor

export
ceil : HasIO io => XlaOp -> io XlaOp
ceil = unaryOp prim__ceil

export
log : HasIO io => XlaOp -> io XlaOp
log = unaryOp prim__log

export
logistic : HasIO io => XlaOp -> io XlaOp
logistic = unaryOp prim__logistic

export
cos : HasIO io => XlaOp -> io XlaOp
cos = unaryOp prim__cos

export
sin : HasIO io => XlaOp -> io XlaOp
sin = unaryOp prim__sin

export
tanh : HasIO io => XlaOp -> io XlaOp
tanh = unaryOp prim__tanh

export
sqrt : HasIO io => XlaOp -> io XlaOp
sqrt = unaryOp prim__sqrt

export
pow : HasIO io => XlaOp -> XlaOp -> io XlaOp
pow = binaryOp prim__pow

export
neg : HasIO io => XlaOp -> io XlaOp
neg = unaryOp prim__neg

export
transpose : HasIO io => XlaOp -> List Nat -> io XlaOp
transpose (MkXlaOp operand) permutation = do
  MkIntArray permutationIntArrayPtr <- mkIntArray permutation
  opPtr <- primIO $ prim__transpose operand permutationIntArrayPtr (cast $ length permutation)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
map : HasIO io => XlaBuilder -> List XlaOp -> XlaComputation -> List Nat -> io XlaOp
map (MkXlaBuilder builder) operands (MkXlaComputation computation) dimensions = do
  MkXlaOpArray operandsXlaOpArrayPtr <- mkXlaOpArray operands
  MkIntArray dimensionsIntArrayPtr <- mkIntArray dimensions
  opPtr <- primIO $ prim__map
    builder
    operandsXlaOpArrayPtr (cast $ length operands)
    computation
    dimensionsIntArrayPtr (cast $ length dimensions)
    prim__getNullAnyPtr 0
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

export
conditional : HasIO io => XlaOp -> XlaOp -> XlaComputation -> XlaOp -> XlaComputation -> io XlaOp
conditional
  (MkXlaOp pred)
  (MkXlaOp trueOperand)
  (MkXlaComputation trueComputation)
  (MkXlaOp falseOperand)
  (MkXlaComputation falseComputation) = do
    opPtr <- primIO $ prim__conditional
      pred
      trueOperand
      trueComputation
      falseOperand
      falseComputation
    opPtr <- onCollectAny opPtr XlaOp.delete
    pure (MkXlaOp opPtr)
