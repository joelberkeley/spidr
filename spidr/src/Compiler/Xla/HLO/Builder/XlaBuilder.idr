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
module Compiler.Xla.HLO.Builder.XlaBuilder

import Compiler.FFI
import Compiler.Xla.HLO.Builder.XlaComputation
import Compiler.Xla.XlaData
import Compiler.Xla.Literal
import Compiler.Xla.Shape
import Compiler.Xla.XlaData
import Types
import Util

public export
data XlaBuilder : Type where
  MkXlaBuilder : GCAnyPtr -> XlaBuilder

public export
data XlaOp : Type where
  MkXlaOp : GCAnyPtr -> XlaOp

namespace XlaBuilder
  %foreign (libxla "XlaBuilder_delete")
  prim__delete : AnyPtr -> PrimIO ()

  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . XlaBuilder.prim__delete

namespace XlaOp
  %foreign (libxla "XlaOp_delete")
  prim__delete : AnyPtr -> PrimIO ()

  export
  delete : HasIO io => AnyPtr -> io ()
  delete = primIO . XlaOp.prim__delete

%foreign (libxla "XlaBuilder_new")
prim__mkXlaBuilder : String -> PrimIO AnyPtr

export
mkXlaBuilder : HasIO io => String -> io XlaBuilder
mkXlaBuilder computationName = do
  ptr <- primIO (prim__mkXlaBuilder computationName)
  ptr <- onCollectAny ptr XlaBuilder.delete
  pure (MkXlaBuilder ptr)

%foreign (libxla "XlaBuilder_name")
prim__XlaBuilder_name : GCAnyPtr -> PrimIO String

export
name : HasIO io => XlaBuilder -> io String
name (MkXlaBuilder builderPtr) = primIO (prim__XlaBuilder_name builderPtr)

%foreign (libxla "CreateSubBuilder")
prim__createSubBuilder : GCAnyPtr -> String -> PrimIO AnyPtr

export
createSubBuilder : HasIO io => XlaBuilder -> String -> io XlaBuilder
createSubBuilder (MkXlaBuilder builderPtr) computationName = do
  subBuilderPtr <- primIO (prim__createSubBuilder builderPtr computationName)
  subBuilderPtr <- onCollectAny subBuilderPtr XlaBuilder.delete
  pure (MkXlaBuilder subBuilderPtr)

%foreign (libxla "XlaBuilder_Build")
prim__build : GCAnyPtr -> GCAnyPtr -> AnyPtr

export
build : HasIO io => XlaBuilder -> XlaOp -> io XlaComputation
build (MkXlaBuilder ptr) (MkXlaOp root)= do
  let computationPtr = prim__build ptr root
  computationPtr <- onCollectAny computationPtr (const $ pure ()) -- XlaComputation.delete
  pure (MkXlaComputation computationPtr)

%foreign (libxla "XlaBuilder_OpToString")
prim__opToString : GCAnyPtr -> GCAnyPtr -> String

export
opToString : XlaBuilder -> XlaOp -> String
opToString (MkXlaBuilder builderPtr) (MkXlaOp opPtr) = prim__opToString builderPtr opPtr

%foreign (libxla "sizeof_XlaOp")
sizeOfXlaOp : Int

%foreign (libxla "set_array_XlaOp")
prim__setArrayXlaOp : AnyPtr -> Int -> GCAnyPtr -> PrimIO ()

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

%foreign (libxla "Parameter")
prim__parameter : GCAnyPtr -> Int -> GCAnyPtr -> String -> PrimIO AnyPtr

export
parameter : HasIO io => XlaBuilder -> Nat -> Xla.Shape -> String -> io XlaOp
parameter (MkXlaBuilder builderPtr) position (MkShape shapePtr) name = do
  opPtr <- primIO $ prim__parameter builderPtr (cast position) shapePtr name
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "ConstantLiteral")
prim__constantLiteral : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
constantLiteral : HasIO io => XlaBuilder -> Literal -> io XlaOp
constantLiteral (MkXlaBuilder builderPtr) (MkLiteral literalPtr) = do
  opPtr <- primIO (prim__constantLiteral builderPtr literalPtr)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Broadcast")
prim__broadcast : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
broadcast : HasIO io => XlaOp -> List Nat -> io XlaOp
broadcast (MkXlaOp opPtr) broadcastSizes = do
  MkIntArray broadcastSizesArrayPtr <- mkIntArray broadcastSizes
  opPtr <- primIO $ prim__broadcast opPtr broadcastSizesArrayPtr (cast $ length broadcastSizes)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "BroadcastInDim")
prim__broadcastInDim : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

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

%foreign (libxla "Reshape")
prim__reshape : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

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

%foreign (libxla "Slice")
prim__slice : GCAnyPtr -> GCPtr Int -> Int -> GCPtr Int -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

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

%foreign (libxla "DynamicSlice")
prim__dynamicSlice : GCAnyPtr -> GCAnyPtr -> Int -> GCPtr Int -> Int -> PrimIO AnyPtr

export
dynamicSlice : HasIO io => XlaOp -> List XlaOp -> List Nat -> io XlaOp
dynamicSlice (MkXlaOp opPtr) startIndices sizeIndices = do
  MkXlaOpArray startIndicesArrayPtr <- mkXlaOpArray startIndices
  MkIntArray sizeIndicesArrayPtr <- mkIntArray sizeIndices
  opPtr <- primIO $ prim__dynamicSlice
    opPtr
    startIndicesArrayPtr (cast $ length startIndices)
    sizeIndicesArrayPtr (cast $ length sizeIndices)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "ConcatInDim")
prim__concatInDim : GCAnyPtr -> GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

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

%foreign (libxla "Select")
prim__select : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
select : HasIO io => XlaOp -> XlaOp -> XlaOp -> io XlaOp
select (MkXlaOp pred) (MkXlaOp onTrue) (MkXlaOp onFalse) = do
  opPtr <- primIO $ prim__select pred onTrue onFalse
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Tuple")
prim__tuple : GCAnyPtr -> GCAnyPtr -> Bits64 -> PrimIO AnyPtr

export
tuple : HasIO io => XlaBuilder -> List XlaOp -> io XlaOp
tuple (MkXlaBuilder builder) elements = do
  MkXlaOpArray xlaOpArrayPtr <- mkXlaOpArray elements
  opPtr <- primIO $ prim__tuple builder xlaOpArrayPtr (cast $ length elements)
  opPtr <- onCollectAny opPtr (const $ pure ()) -- XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "GetTupleElement")
prim__getTupleElement : GCAnyPtr -> Bits64 -> PrimIO AnyPtr

export
getTupleElement : HasIO io => XlaOp -> Nat -> io XlaOp
getTupleElement (MkXlaOp tuple_) index = do
  opPtr <- primIO $ prim__getTupleElement tuple_ (cast index)
  opPtr <- onCollectAny opPtr (const $ pure ()) -- XlaOp.delete
  pure (MkXlaOp opPtr)

binaryOp : HasIO io => (GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr) -> XlaOp -> XlaOp -> io XlaOp
binaryOp prim__f (MkXlaOp x) (MkXlaOp y) = do
  opPtr <- primIO $ prim__f x y
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Eq")
prim__eq : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
eq : HasIO io => XlaOp -> XlaOp -> io XlaOp
eq = binaryOp prim__eq

%foreign (libxla "Ne")
prim__ne : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
ne : HasIO io => XlaOp -> XlaOp -> io XlaOp
ne = binaryOp prim__ne

%foreign (libxla "Ge")
prim__ge : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
ge : HasIO io => XlaOp -> XlaOp -> io XlaOp
ge = binaryOp prim__ge

%foreign (libxla "Gt")
prim__gt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
gt : HasIO io => XlaOp -> XlaOp -> io XlaOp
gt = binaryOp prim__gt

%foreign (libxla "Lt")
prim__lt : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
lt : HasIO io => XlaOp -> XlaOp -> io XlaOp
lt = binaryOp prim__lt

%foreign (libxla "Le")
prim__le : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
le : HasIO io => XlaOp -> XlaOp -> io XlaOp
le = binaryOp prim__le

%foreign (libxla "Dot")
prim__dot : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
dot : HasIO io => XlaOp -> XlaOp -> io XlaOp
dot = binaryOp prim__dot

%foreign (libxla "DotGeneral")
prim__dotGeneral : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
dotGeneral : HasIO io => XlaOp -> XlaOp -> DotDimensionNumbers -> io XlaOp
dotGeneral (MkXlaOp l) (MkXlaOp r) (MkDotDimensionNumbers dimensionNumbers) = do
  opPtr <- primIO $ prim__dotGeneral l r dimensionNumbers
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "TriangularSolve")
prim__triangularSolve : GCAnyPtr -> GCAnyPtr -> Int -> Int -> Int -> Int -> PrimIO AnyPtr

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

%foreign (libxla "Cholesky")
prim__cholesky : GCAnyPtr -> Int -> PrimIO AnyPtr

export
cholesky : HasIO io => XlaOp -> Bool -> io XlaOp
cholesky (MkXlaOp a) lower = do
  opPtr <- primIO $ prim__cholesky a (boolToCInt lower)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Add")
prim__add : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
add : HasIO io => XlaOp -> XlaOp -> io XlaOp
add = binaryOp prim__add

%foreign (libxla "Sub")
prim__sub : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
sub : HasIO io => XlaOp -> XlaOp -> io XlaOp
sub = binaryOp prim__sub

%foreign (libxla "Mul")
prim__mul : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
mul : HasIO io => XlaOp -> XlaOp -> io XlaOp
mul = binaryOp prim__mul

%foreign (libxla "Div")
prim__div : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
div : HasIO io => XlaOp -> XlaOp -> io XlaOp
div = binaryOp prim__div

%foreign (libxla "Rem")
prim__rem : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
rem : HasIO io => XlaOp -> XlaOp -> io XlaOp
rem = binaryOp prim__rem

%foreign (libxla "Max")
prim__max : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
max : HasIO io => XlaOp -> XlaOp -> io XlaOp
max = binaryOp prim__max

%foreign (libxla "Min")
prim__min : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
min : HasIO io => XlaOp -> XlaOp -> io XlaOp
min = binaryOp prim__min

%foreign (libxla "And")
prim__and : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
and : HasIO io => XlaOp -> XlaOp -> io XlaOp
and = binaryOp prim__and

%foreign (libxla "Or")
prim__or : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
or : HasIO io => XlaOp -> XlaOp -> io XlaOp
or = binaryOp prim__or

export
unaryOp : HasIO io => (GCAnyPtr -> PrimIO AnyPtr) -> XlaOp -> io XlaOp
unaryOp prim__f (MkXlaOp x) = do
  opPtr <- primIO $ prim__f x
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Not")
prim__not : GCAnyPtr -> PrimIO AnyPtr

export
not : HasIO io => XlaOp -> io XlaOp
not = unaryOp prim__not

%foreign (libxla "Reduce")
prim__reduce : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
reduce : HasIO io => XlaOp -> XlaOp -> XlaComputation -> List Nat -> io XlaOp
reduce (MkXlaOp operand) (MkXlaOp initValue) (MkXlaComputation computation) dimensions = do
  MkIntArray dimensionsIntArrayPtr <- mkIntArray dimensions
  opPtr <- primIO $ prim__reduce
    operand initValue computation dimensionsIntArrayPtr (cast $ length dimensions)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Abs")
prim__abs : GCAnyPtr -> PrimIO AnyPtr

export
abs : HasIO io => XlaOp -> io XlaOp
abs = unaryOp prim__abs

%foreign (libxla "Exp")
prim__exp : GCAnyPtr -> PrimIO AnyPtr

export
exp : HasIO io => XlaOp -> io XlaOp
exp = unaryOp prim__exp

%foreign (libxla "Floor")
prim__floor : GCAnyPtr -> PrimIO AnyPtr

export
floor : HasIO io => XlaOp -> io XlaOp
floor = unaryOp prim__floor

%foreign (libxla "Ceil")
prim__ceil : GCAnyPtr -> PrimIO AnyPtr

export
ceil : HasIO io => XlaOp -> io XlaOp
ceil = unaryOp prim__ceil

%foreign (libxla "Log")
prim__log : GCAnyPtr -> PrimIO AnyPtr

export
log : HasIO io => XlaOp -> io XlaOp
log = unaryOp prim__log

%foreign (libxla "Logistic")
prim__logistic : GCAnyPtr -> PrimIO AnyPtr

export
logistic : HasIO io => XlaOp -> io XlaOp
logistic = unaryOp prim__logistic

%foreign (libxla "Cos")
prim__cos : GCAnyPtr -> PrimIO AnyPtr

export
cos : HasIO io => XlaOp -> io XlaOp
cos = unaryOp prim__cos

%foreign (libxla "Sin")
prim__sin : GCAnyPtr -> PrimIO AnyPtr

export
sin : HasIO io => XlaOp -> io XlaOp
sin = unaryOp prim__sin

%foreign (libxla "Tanh")
prim__tanh : GCAnyPtr -> PrimIO AnyPtr

export
tanh : HasIO io => XlaOp -> io XlaOp
tanh = unaryOp prim__tanh

%foreign (libxla "Sqrt")
prim__sqrt : GCAnyPtr -> PrimIO AnyPtr

export
sqrt : HasIO io => XlaOp -> io XlaOp
sqrt = unaryOp prim__sqrt

%foreign (libxla "Pow")
prim__pow : GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
pow : HasIO io => XlaOp -> XlaOp -> io XlaOp
pow = binaryOp prim__pow

%foreign (libxla "Iota")
prim__iota : GCAnyPtr -> GCAnyPtr -> Int -> PrimIO AnyPtr

export
iota : HasIO io => XlaBuilder -> Xla.Shape -> Nat -> io XlaOp
iota (MkXlaBuilder xlaBuilder) (MkShape shape) iota_dimension = do
  opPtr <- primIO $ prim__iota xlaBuilder shape (cast iota_dimension)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "ConvertElementType")
prim__convertElementType : GCAnyPtr -> Int -> PrimIO AnyPtr

export
convertElementType : (HasIO io, Primitive dtype) => XlaOp -> io XlaOp
convertElementType (MkXlaOp operand) = do
  opPtr <- primIO $ prim__convertElementType operand (xlaIdentifier {dtype})
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Neg")
prim__neg : GCAnyPtr -> PrimIO AnyPtr

export
neg : HasIO io => XlaOp -> io XlaOp
neg = unaryOp prim__neg

%foreign (libxla "Transpose")
prim__transpose : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
transpose : HasIO io => XlaOp -> List Nat -> io XlaOp
transpose (MkXlaOp operand) permutation = do
  MkIntArray permutationIntArrayPtr <- mkIntArray permutation
  opPtr <- primIO $ prim__transpose operand permutationIntArrayPtr (cast $ length permutation)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Rev")
prim__rev : GCAnyPtr -> GCPtr Int -> Int -> PrimIO AnyPtr

export
rev : HasIO io => XlaOp -> List Nat -> io XlaOp
rev (MkXlaOp operand) dimensions = do
  MkIntArray dimensionsIntArrayPtr <- mkIntArray dimensions
  opPtr <- primIO $ prim__rev operand dimensionsIntArrayPtr (cast $ length dimensions)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Sort")
prim__sort : GCAnyPtr -> Int -> GCAnyPtr -> Int -> Int -> PrimIO AnyPtr

export
sort : HasIO io => List XlaOp -> XlaComputation -> Nat -> Bool -> io XlaOp
sort operands (MkXlaComputation comparator) dimension isStable = do
  MkXlaOpArray operandsXlaOpArrayPtr <- mkXlaOpArray operands
  opPtr <- primIO $ prim__sort
    operandsXlaOpArrayPtr (cast $ length operands)
    comparator
    (cast dimension)
    (boolToCInt isStable)
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Map")
prim__map :
  GCAnyPtr -> GCAnyPtr -> Int -> GCAnyPtr -> GCPtr Int -> Int -> AnyPtr -> Int -> PrimIO AnyPtr

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

public export
data RandomAlgorithm = RngDefault | RngThreeFry | RngPhilox

%foreign (libxla "RngBitGenerator")
prim__rngBitGenerator : Int -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
rngBitGenerator : HasIO io => RandomAlgorithm -> XlaOp -> Xla.Shape -> io XlaOp
rngBitGenerator algorithm (MkXlaOp initialState) (MkShape shape) = do
  let algorithm : Int = case algorithm of
        RngDefault => 0
        RngThreeFry => 1
        RngPhilox => 2
  opPtr <- primIO $ prim__rngBitGenerator algorithm initialState shape
  opPtr <- onCollectAny opPtr XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "While")
prim__while : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

export
while : HasIO io => XlaComputation -> XlaComputation -> XlaOp -> io XlaOp
while (MkXlaComputation condition) (MkXlaComputation body) (MkXlaOp init) = do
  opPtr <- primIO $ prim__while condition body init
  opPtr <- onCollectAny opPtr (const $ pure ()) -- XlaOp.delete
  pure (MkXlaOp opPtr)

%foreign (libxla "Conditional")
prim__conditional : GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> GCAnyPtr -> PrimIO AnyPtr

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
