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
||| This module contains the `Tensor`, an array of numbers or booleans, along with a
||| number of functions operating on `Tensor`s. spidr tracks tensor shape and data type in the
||| types, so you can be sure that if your tensor code compiles, the shapes and types are
||| consistent.
|||
||| spidr achieves efficient reuse of tensor computations with `Graph`. See the tutorial
||| _The Graph Compiler_ for a discussion of pitfalls to avoid when using `Graph`.
module Tensor

import Control.Monad.Error.Either
import public Control.Monad.State
import public Data.List
import public Data.List.Elem
import Data.List.Quantifiers
import Decidable.Equality

import Compiler.Eval
import Compiler.Expr
import Compiler.Xla.Shape
import Compiler.Xla.ShapeUtil
import Compiler.LiteralRW
import Device
import Literal
import public Primitive
import public Types
import public Util

0 XlaShape : Type
XlaShape = Xla.Shape

%hide Xla.Shape

----------------------------- core definitions ----------------------------

||| A scalar or array. Construct a `Tensor` with the function `tensor`.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (shape : Shape) -> (dtype : Type) -> Type where
  MkTensor : Expr -> {shape : _} -> Tensor shape dtype

||| The effect of labelling nodes in a computational graph.
export
data GraphT : (Type -> Type) -> Type -> Type where
  MkGraphT : StateT Env m a -> GraphT m a

public export 0
Graph : Type -> Type
Graph = GraphT Identity

export
Functor m => Functor (GraphT m) where
  map f (MkGraphT x) = MkGraphT (map f x)

export
Monad m => Applicative (GraphT m) where
  pure x = MkGraphT (pure x)
  (MkGraphT f) <*> (MkGraphT x) = MkGraphT (f <*> x)

export
Monad m => Monad (GraphT m) where
  (MkGraphT x) >>= f = MkGraphT $ x >>= (\y => let MkGraphT z = f y in z)

export
MonadTrans GraphT where
  lift = MkGraphT . lift

public export
interface Shareable a where
  ||| Mark an expression to be efficiently reused. For example, in
  ||| ```
  ||| bad : Tensor [9999999] F64
  ||| bad = let x = fill {shape = [9999999]} 1.0 in x + x
  |||
  ||| good : Graph $ Tensor [9999999] F64
  ||| good = do x <- share $ fill {shape = [9999999]} 1.0
  |||           pure (x + x)
  ||| ```
  ||| the large vector `x` is calculated twice in `bad`, but once in `good`, as `share` marks it for sharing.
  |||
  ||| Types that implement this interface should `share` constituent components it deems worth sharing.
  ||| For example, see the implementation for tuples.
  |||
  ||| See tutorial [_Nuisances in the Tensor API_](https://github.com/joelberkeley/spidr/blob/master/tutorials/Nuisances.md) for details.
  share : a -> Graph a

export
Shareable (Tensor shape dtype) where
  share x@(MkTensor (Var _)) = pure x  -- not necessary, but saves space
  share (MkTensor x) = MkGraphT $ do
    x <- shareExpr x
    pure $ MkTensor x

export
(Shareable a, Shareable b) => Shareable (a, b) where
  share (a, b) = [| (share a, share b) |]

||| Construct a `Tensor` from `Literal` data. For example
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[1, 2, 3],
|||             [4, 5, 6]]
||| ```
export
tensor : PrimitiveRW dtype a => {shape : _} -> Literal shape a -> Tensor shape dtype
tensor lit = MkTensor $ FromLiteral {dtype} {shape} lit

namespace F64
  export
  fromDouble : Double -> Tensor [] F64
  fromDouble = tensor . Scalar

namespace S32
  export
  fromInteger : Integer -> Tensor [] S32
  fromInteger = tensor . Scalar . fromInteger

partial
try : Show e => EitherT e IO a -> IO a
try = eitherT (idris_crash . show) pure

namespace Graph
  ||| Evaluate a `Tensor`, returning its value as a `Literal`. This function builds and executes the
  ||| computational graph.
  |||
  ||| **Note:** Each call to `eval` will rebuild and execute the graph; multiple calls to `eval` on
  ||| different tensors, even if they are in the same computation, will be treated independently.
  ||| To efficiently evaluate multiple tensors at once, use `TensorList.Graph.eval`.
  export partial
  eval : Device -> PrimitiveRW dtype ty => Graph (Tensor shape dtype) -> IO (Literal shape ty)
  eval device (MkGraphT x) =
    let (env, MkTensor root) = runState empty x
     in try $ do
          shape <- mkShape shape {dtype}
          [lit] <- execute device (MkFn [] root env) [shape]
          read {dtype} [] lit

||| A convenience wrapper for `Graph.eval`, for use with a bare `Tensor`.
export partial
eval : Device -> PrimitiveRW dtype ty => Tensor shape dtype -> IO (Literal shape ty)
eval device x = eval device (pure x)

namespace TensorList
  namespace Graph
    ||| A list of `Tensor`s, along with the conversions needed to evaluate them to `Literal`s.
    ||| The list is parametrized by the shapes and types of the resulting `Literal`s.
    public export
    data TensorList : List Shape -> List Type -> Type where
      Nil : TensorList [] []
      (::) : PrimitiveRW dtype ty =>
             Tensor shape dtype ->
             TensorList shapes tys ->
             TensorList (shape :: shapes) (ty :: tys)

    ||| Evaluate a list of `Tensor`s as a list of `Literal`s. Tensors in the list can have different
    ||| shapes and element types. For example,
    ||| ```
    ||| main : Device -> IO ()
    ||| main device = do [x, y] <- eval device $ do let x = tensor {dtype = F64} [1.2, 3.4]
    |||                                             y <- reduce @{Sum} [0] x
    |||                                             pure [x, y]
    |||                  printLn x
    |||                  printLn y
    ||| ```
    ||| In contrast to `Tensor.eval` when called on multiple tensors, this function constructs and
    ||| compiles the graph just once.
    export partial
    eval : Device -> Graph (TensorList shapes tys) -> IO (All2 Literal shapes tys)
    eval device (MkGraphT xs) =
      let (env, xs) = runState empty xs
          root = Tuple $ nodes xs
       in try $ do
            xlaShapes <- buildShapes xs
            let (outputs ** eq) = lengthC xs
            lits <- execute device (MkFn [] root env) {outputs} (rewrite eq in xlaShapes)
            readAll xs $ rewrite sym eq in lits

      where

      lengthC : TensorList s t -> (n ** n === length s)
      lengthC [] = (0 ** Refl)
      lengthC (_ :: xs) = let (n ** eq) = lengthC xs in (S n ** cong S eq)

      buildShapes : HasIO io => TensorList s t -> io $ Vect (length s) XlaShape
      buildShapes [] = pure []
      buildShapes (MkTensor {shape, dtype} _ :: ts) = [| mkShape shape {dtype} :: buildShapes ts |]

      nodes : TensorList s t -> List Expr
      nodes [] = []
      nodes (MkTensor x :: xs) = x :: nodes xs

      readAll : HasIO io => TensorList s t -> Vect (length s) Literal -> io $ All2 Literal s t
      readAll [] _ = pure []
      readAll (MkTensor {dtype} _ :: ts) (l :: ls) = [| read {dtype} [] l :: readAll ts ls |]

  ||| A convenience wrapper for `TensorList.Graph.eval`, for use with a bare `TensorList`.
  export partial
  eval : Device -> TensorList shapes tys -> IO (All2 Literal shapes tys)
  eval device xs = eval device (pure xs)

||| A string representation of a tensor graph.
|||
||| There are no guarantees whatsoever as to the string structure and contents.
export
Show (Graph $ Tensor shape dtype) where
  show (MkGraphT x) = let (env, MkTensor root) = runState empty x in show (MkFn [] root env)

||| Bounds for numeric tensors. Will be infinite for floating point types.
export
[NonFinite] Primitive.Ord dtype => Bounded (Tensor [] dtype) where
  min = MkTensor $ MinValue {dtype}
  max = MkTensor $ MaxValue {dtype}

||| Finite bounds for numeric tensors.
export
[Finite] Primitive.Ord dtype => Bounded (Tensor [] dtype) where
  min = MkTensor $ MinFiniteValue {dtype}
  max = MkTensor $ MaxFiniteValue {dtype}

||| Cast the element type. For example, `castDtype (tensor {dtype = S32} [1, -2])` is
||| `tensor {dtype = F64} [1.0, -2.0]`.
export
castDtype : Primitive.Integral a => Tensor shape a -> Tensor shape F64
castDtype $ MkTensor x = MkTensor $ ConvertElementType {dtype = F64} x

----------------------------- structural operations ----------------------------

||| Reshape a `Tensor`. For example, `reshape {to = [2, 1]} (tensor [3, 4])` is
||| `tensor [[3], [4]]`. The output can have a different rank to the input.
export
reshape :
  Primitive dtype =>
  {to : _} ->
  {auto 0 sizesEqual : product from = product to} ->
  Tensor from dtype ->
  Tensor to dtype
reshape $ MkTensor {shape} x = MkTensor $ Reshape shape to x

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified `axis` in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ tensor [[1, 2], [3, 4], [5, 6]]` is `tensor [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand :
  Primitive dtype =>
  (axis : Nat) ->
  {auto 0 inBounds : axis `LTE` length shape} ->
  Tensor shape dtype ->
  Tensor (insertAt axis 1 shape) dtype
expand axis $ MkTensor {shape = _} x = MkTensor $ Reshape shape (insertAt axis 1 shape) x

namespace Squeezable
  ||| A `Squeezable from to` constitutes proof that the shape `from` can be squeezed to the
  ||| shape `to`. Squeezing is the process of removing any number of dimensions of length one.
  public export
  data Squeezable : (0 from : Shape) -> (0 to : Shape) -> Type where
    ||| Proof that a shape can be squeezed to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    Same : Squeezable x x

    ||| Proof that any dimensions (including those of length 1) can be preserved in the process of
    ||| squeezing. For example:
    |||
    ||| ...
    Match : Squeezable from to -> Squeezable (x :: from) (x :: to)

    ||| Proof that any dimensions of length one can be squeezed out. For example:
    |||
    ||| [1, 3, 1, 1, 4] to [3, 4]
    Nest : Squeezable from to -> Squeezable (1 :: from) to

||| Remove dimensions of length one from a `Tensor` such that it has the desired shape. For example:
|||
||| ```
||| x : Tensor [2, 1, 3, 1] S32
||| x = tensor [[[[4], [5], [6]]],
|||             [[[7], [8], [9]]]]
|||
||| y : Tensor [2, 1, 3] S32
||| y = squeeze x
||| ```
||| is
||| ```
||| y : Tensor [2, 1, 3] S32
||| y = tensor [[[4, 5, 6]],
|||             [[7, 8, 9]]]
||| ```
export
squeeze :
  Primitive dtype =>
  {to : _} ->
  {auto 0 shapesSqueezable : Squeezable from to} ->
  Tensor from dtype ->
  Tensor to dtype
squeeze $ MkTensor {shape} x = MkTensor $ Reshape shape to x

||| A `SliceOrIndex d` is a valid slice or index into a dimension of size `d`. See `slice` for
||| details.
export
data SliceOrIndex : Nat -> Type where
  Slice :
    (from, to : Nat) ->
    {size : _} ->
    {auto 0 fromTo : from + size = to} ->
    {auto 0 inDim : LTE to d} ->
    SliceOrIndex d
  Index : (idx : Nat) -> {auto 0 inDim : LT idx d} -> SliceOrIndex d
  DynamicSlice : Tensor [] U64 -> (size : Nat) -> {auto 0 inDim : LTE size d} -> SliceOrIndex d
  DynamicIndex : Tensor [] U64 -> SliceOrIndex d

||| Index at `idx`. See `slice` for details.
public export
at : (idx : Nat) -> {auto 0 inDim : LT idx d} -> SliceOrIndex d
at = Index

namespace Dynamic
  ||| Index at the specified index. See `slice` for details.
  public export
  at : Tensor [] U64 -> SliceOrIndex d
  at = DynamicIndex

||| Slice from `from` (inclusive) to `to` (exclusive). See `slice` for details.
public export
(.to) :
  (from, to : Nat) ->
  {size : _} ->
  {auto 0 fromTo : from + size = to} ->
  {auto 0 inDim : LTE to d} ->
  SliceOrIndex d
(.to) = Slice

||| Slice `size` elements starting at the specified scalar `U64` index. See `slice` for details.
public export
(.size) : Tensor [] U64 -> (size : Nat) -> {auto 0 inDim : LTE size d} -> SliceOrIndex d
(.size) = DynamicSlice

||| Slice across all indices along an axis. See `slice` for details.
public export
all : {d : _} -> SliceOrIndex d
all = Slice 0 @{%search} @{reflexive {ty = Nat}} d

||| A `MultiSlice shape` is a valid multi-dimensional slice into a tensor with shape `shape`.
||| See `slice` for details.
public export
data MultiSlice : Shape -> Type where
  Nil : MultiSlice ds
  (::) : SliceOrIndex d -> MultiSlice ds -> MultiSlice (d :: ds)

namespace MultiSlice
  ||| The shape of a tensor produced by slicing with the specified multi-dimensional slice. See
  ||| `Tensor.slice` for details.
  public export
  slice : {shape : _} -> MultiSlice shape -> Shape
  slice {shape} [] = shape
  slice {shape = (_ :: _)} (Slice {size} _ _ :: xs) = size :: slice xs
  slice {shape = (_ :: _)} (Index _ :: xs) = slice xs
  slice {shape = (_ :: _)} (DynamicSlice _ size :: xs) = size :: slice xs
  slice {shape = (_ :: _)} (DynamicIndex _ :: xs) = slice xs

||| Slice or index `Tensor` axes. Each axis can be sliced or indexed, and this can be done with
||| either static (`Nat`) or dynamic (scalar `U64`) indices.
|||
||| **Static indices**
|||
||| Static indices are `Nat`s. For example, for
||| ```
||| x : Tensor [5, 6] S32
||| x = tensor [[ 0,  1,  2,  3,  4,  5],
|||             [ 6,  7,  8,  9, 10, 11],
|||             [12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23],
|||             [24, 25, 26, 27, 28, 29]]
||| ```
||| we can index as `slice [at 1] x` to get
||| ```
||| x : Tensor [6] S32
||| x = tensor [6, 7, 8, 9, 10, 11]
||| ```
||| or we can slice as `slice [2.to 4] x` to get
||| ```
||| x : Tensor [2, 6] S32
||| x = tensor [[12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23]]
||| ```
||| Note that in `2.to 4`, the 2 is inclusive, and the 4 exclusive, so we return indices 2 and 3.
|||
||| **Dynamic indices**
|||
||| Dynamic indices are scalar `U64` values, and the API works slightly differently because we
||| can't know the value of dynamic indices until the graph is executed. For indexing, with scalar
||| `U64` index `i` in `slice [at i] x`, `i` is clamped to be a valid index into that dimension.
||| For example, for `i = tensor 1`, `slice [at i] x` is
||| ```
||| x : Tensor [6] S32
||| x = tensor [6, 7, 8, 9, 10, 11]
||| ```
||| as in the static case. However, for `i = tensor 10`, `slice [at i] x` returns the last row
||| ```
||| x : Tensor [6] S32
||| x = tensor [24, 25, 26, 27, 28, 29]
||| ```
||| We can also slice by specifying a scalar `U64` start index, and a static size, as
||| `slice [i.size 2] x` with `i = tensor 2` to get
||| ```
||| x : Tensor [2, 6] S32
||| x = tensor [[12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23]]
||| ```
||| For a given slice `size`, the dynamic start index is clamped such that we always get `size`
||| elements along that axis. For example, `slice [i.size 2] x` with `i = tensor 4` is
||| ```
||| x : Tensor [2, 6] S32
||| x = tensor [[18, 19, 20, 21, 22, 23],
|||             [24, 25, 26, 27, 28, 29]]
||| ```
||| which starts at index 3 rather than index 4.
|||
||| **Mixed static, dynamic, slicing and indexing**
|||
||| Each axis can only be sliced or indexed, and must use only static or dynamic indices. However,
||| across axes, we can mix these four arbitrarily. For example, with `slice [2.to 4, at 1] x` to
||| get
||| ```
||| x : Tensor [2] S32
||| x = tensor [13, 19]
||| ```
||| or with `i = tensor 2` in `slice [at 1, i.size 2] x` to get
||| ```
||| x : Tensor [2] S32
||| x = tensor [7, 8]
||| ```
|||
||| Slices and indices apply to the leading axes of the tensor. For trailing axes omitted from the
||| multi-dimensional slice, the whole of the axis is returned. If we want to slice or index over
||| later axes and retain all indices in a leading axis, we can use the convenience function `all`,
||| as `slice [all, at 3] x` to get
||| ```
||| x : Tensor [5] S32
||| x = tensor [[3], [9], [15], [21], [27]]
||| ```
||| This is exactly the same as the more manual `slice [0.to 5, at 3] x` and
||| `slice [(tensor 0).size 5, at 3] x`.
|||
||| @at The multi-dimensional slices and indices at which to slice the tensor.
export
slice :
  Primitive dtype =>
  (at : MultiSlice shape) ->
  Tensor shape dtype ->
  Tensor (slice at) dtype
slice at $ MkTensor x = MkTensor
  $ Reshape (mapd size id at) (MultiSlice.slice at)
    $ DynamicSlice (dynStarts [] at) (mapd size id at)
      $ Slice (mapd start (const 0) at) (mapd stop id at) (replicate (length shape) 1) x

      where
      mapd : ((Nat -> a) -> {d : Nat} -> SliceOrIndex d -> a) ->
             (Nat -> a) ->
             {shape : Shape} ->
             MultiSlice shape ->
             List a
      mapd _ dflt {shape} [] = Prelude.map dflt shape
      mapd f dflt (x :: xs) = f dflt x :: mapd f dflt xs

      start : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
      start _ (Slice from _) = from
      start _ (Index idx) = idx
      start f {d} _ = f d

      stop : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
      stop _ (Slice _ to) = to
      stop _ (Index idx) = S idx
      stop f {d} _ = f d

      size : (Nat -> Nat) -> {d : Nat} -> SliceOrIndex d -> Nat
      size _ (Slice {size = size'} _ _) = size'
      size _ (Index _) = 1
      size _ (DynamicSlice _ size') = size'
      size _ (DynamicIndex _) = 1

      zero : Expr
      zero = FromLiteral {shape = []} {dtype = U64} 0

      dynStarts : List Expr -> {shape : _} -> MultiSlice shape -> List Expr
      dynStarts idxs {shape} [] = replicate (length shape) zero ++ idxs
      dynStarts idxs (DynamicSlice (MkTensor i) _ :: ds) = i :: dynStarts idxs ds
      dynStarts idxs (DynamicIndex (MkTensor i) :: ds) = i :: dynStarts idxs ds
      dynStarts idxs (_ :: ds) = zero :: dynStarts idxs ds

||| Concatenate two `Tensor`s along the specfied `axis`. For example,
||| `concat 0 (tensor [[1, 2], [3, 4]]) (tensor [[5, 6]])` and
||| `concat 1 (tensor [[3], [6]]) (tensor [[4, 5], [7, 8]])` are both
||| `tensor [[1, 2], [3, 4], [5, 6]]`.
export
concat :
  Primitive dtype =>
  (axis : Nat) ->
  Tensor s dtype ->
  Tensor s' dtype ->
  {auto 0 inBounds : (InBounds axis s, InBounds axis s')} ->
  {auto 0 shapesConcatenable : deleteAt axis s = deleteAt axis s'} ->
  Tensor (replaceAt axis (index axis s + index axis s') s) dtype
concat axis (MkTensor x) (MkTensor x') = MkTensor $ Concat axis x x'

||| The diagonal of a matrix as a vector. For example, for
||| ```
||| x : Tensor [3, 3] S32
||| x = tensor [[0, 1, 2],
|||             [3, 4, 5],
|||             [6, 7, 8]]
||| ```
||| `diag x` is `tensor [0, 4, 8]`.
export
diag : Primitive dtype => Tensor [n, n] dtype -> Tensor [n] dtype
diag $ MkTensor x = MkTensor $ Diag x

||| Represents the upper- or lower-triangular component of a matrix.
public export
data Triangle = Upper | Lower

||| Get the upper- or lower-triangular component of a matrix. For example, for
||| ```
||| x : Tensor [3, 3] S32
||| x = tensor [[1, 2, 3],
|||             [4, 5, 6],
|||             [7, 8, 9]]
||| ```
||| `triangle Lower x` is
||| ```
||| x : Tensor [3, 3] S32
||| x = tensor [[1, 0, 0],
|||             [4, 5, 0],
|||             [7, 8, 9]]
||| ```
export
triangle : Primitive dtype => Triangle -> Tensor [n, n] dtype -> Tensor [n, n] dtype
triangle tri $ MkTensor x = MkTensor $ Triangle (case tri of Upper => False; Lower => True) x

||| Transpose a matrix. For example, `(tensor [[1, 2], [3, 4]]).T` is `tensor [[1, 3], [2, 4]]`.
export
(.T) : Tensor [m, n] dtype -> Tensor [n, m] dtype
(MkTensor x).T = MkTensor $ Transpose [1, 0] x

||| Transpose axes of a tensor. This is a more general version of `(.T)`, in which you can
||| transpose any number of axes in a tensor of arbitrary rank. The i'th axis in the resulting
||| tensor corresponds to the `index i ordering`'th axis in the input tensor. For example, for
||| ```
||| x : Tensor [2, 3, 4] S32
||| x = tensor [[[ 0,  1,  2,  3],
|||              [ 4,  5,  6,  7],
|||              [ 8,  9, 10, 11]],
|||             [[12, 13, 14, 15],
|||              [16, 17, 18, 19],
|||              [20, 21, 22, 23]]]
||| ```
||| `transpose [0, 2, 1] x` is
||| ```
||| x : Tensor [2, 4, 3] S32
||| x = tensor [[[ 0,  4,  8],
|||              [ 1,  5,  9],
|||              [ 2,  6, 10],
|||              [ 3,  7, 11]],
|||             [[12, 16, 20],
|||              [13, 17, 21],
|||              [14, 18, 22],
|||              [15, 19, 23]]]
||| ```
||| `transpose [2, 0, 1] x` is
||| ```
||| x : Tensor [4, 2, 3] S32
||| x = tensor [[[ 0,  4,  8],
|||              [12, 16, 20]],
|||             [[ 1,  5,  9],
|||              [13, 17, 21]],
|||             [[ 2,  6, 10],
|||              [14, 18, 22]],
|||             [[ 3,  7, 11],
|||              [15, 19, 23]]]
||| ```
|||
||| In order to see what effect transposing a tensor has, it can help to bear in mind the following:
||| * if an element can be found with `slice [at 3, at 4, at 5] x` in the original tensor,
|||   that same element can instead be found with `slice [at 5, at 3, at 4]` given a
|||   `transpose [2, 0, 1]`. That is, transposing axes re-orders indices when indexing.
||| * with `transpose [2, 0, 1]`, traversing the first axis in the result is equivalent to
|||   traversing the last axis in the input. Similarly, traversing the last axis in the result is
|||   equivalent to traversing the second axis in the input.
export
transpose :
  (ordering : List Nat) ->
  Tensor shape dtype ->
  {auto 0 lengths : length ordering = length shape} ->
  {auto 0 axesUnique : unique ordering = True} ->
  {auto 0 inBounds : All (flip InBounds shape) ordering} ->
  Tensor (multiIndex ordering shape) dtype
transpose ordering $ MkTensor x = MkTensor $ Transpose ordering x

||| The identity tensor, with inferred shape and element type. For example,
||| ```
||| x : Tensor [2, 2] S32
||| x = identity
||| ```
||| is
||| ```
||| x : Tensor [2, 2] S32
||| x = tensor [[1, 0],
|||             [0, 1]]
||| ```
export
identity : Primitive.Num dtype => {n : _} -> Tensor [n, n] dtype
identity = MkTensor $ Identity {dtype} n

||| A `DimBroadcastable from to` proves that a dimension of size `from` can be broadcast to a
||| dimension of size `to`.
public export
data DimBroadcastable : (0 from : Nat) -> (0 to : Nat) -> Type where
  ||| Proof that any dimension can be broadcast to itself. For example in shapes `[2, 3]` to
  ||| `[2, 3]`.
  Same : DimBroadcastable x x

  ||| Proof that a dimension of length one can be broadcast to any size. For example in shapes
  ||| `[2, 1]` to `[2, 3]`
  Stack : DimBroadcastable 1 _

  ||| Proof that any dimension can be broadcast to zero. For example in shapes `[2, 3]` to `[2, 0]`.
  Zero : DimBroadcastable _ 0

namespace Broadcastable
  ||| A `Broadcastable from to` constitutes proof that the shape `from` can be broadcast to the
  ||| shape `to`.
  public export
  data Broadcastable : (0 from : Shape) -> (0 to : Shape) -> Type where
    ||| Proof that a shape can be broadcast to itself. For example:
    |||
    ||| [] to []
    ||| [3, 4] to [3, 4]
    |||
    ||| Implementation note: we could have used `Broadcast [] []`, which would have resulted in more
    ||| atomic constructors for `Broadcastable`, but the author guesses that this implementation helps
    ||| the type checker avoid applications of `Match`.
    Same : Broadcastable x x

    ||| Proof that a dimension of size `f` can be broadcast to size `t` if these dimensions
    ||| are `DimBroadcastable f t`. For example:
    |||
    ||| [2, 3] to [2, 3]
    ||| [2, 1] to [2, 3]
    ||| [2, 1] to [2, 0]
    Match : forall from, to .
            {auto 0 ranksEq : length from = length to} ->
            {auto 0 dimBroadcastable : DimBroadcastable f t} ->
            Broadcastable from to ->
            Broadcastable (f :: from) (t :: to)

    ||| Proof that broadcasting can add outer dimensions i.e. nesting. For example:
    |||
    ||| [3] to [1, 3]
    ||| [3] to [5, 3]
    Nest : Broadcastable f t -> Broadcastable f (_ :: t)

||| A shape can be extended with any number of leading dimensions.
|||
||| @leading The leading dimensions.
export
broadcastableByLeading : (leading : List Nat) -> Broadcastable shape (leading ++ shape)
broadcastableByLeading [] = Same
broadcastableByLeading (l :: ls) = Nest (broadcastableByLeading ls)

||| A scalar can be broadcast to any shape.
%hint
export
scalarToAnyOk : (to : Shape) -> Broadcastable [] to
scalarToAnyOk to = rewrite sym $ appendNilRightNeutral to in broadcastableByLeading to

||| Broadcast a `Tensor` to a new compatible shape. For example,
||| ```
||| x : Tensor [2, 3] S32
||| x = broadcast (tensor [4, 5, 6])
||| ```
||| is
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[4, 5, 6], [4, 5, 6]]
||| ```
export
broadcast :
  Primitive dtype =>
  {to : _} ->
  {auto shapesOK : Broadcastable from to} ->
  Tensor from dtype ->
  Tensor to dtype
broadcast $ MkTensor {shape = _} x = MkTensor $ Broadcast {dtype} from to x

||| A `Tensor` where every element has the specified value. For example,
||| ```
||| fives : Tensor [2, 3] S32
||| fives = fill 5
||| ```
||| is
||| ```
||| fives : Tensor [2, 3] S32
||| fives = tensor [[5, 5, 5],
|||                 [5, 5, 5]]
||| ```
export
fill : PrimitiveRW dtype ty => {shape : _} -> ty -> Tensor shape dtype
fill x = broadcast {shapesOK = scalarToAnyOk shape} (tensor (Scalar x))

||| A constant where values increment from zero along the specified `axis`. For example,
||| ```
||| x : Tensor [3, 5] S32
||| x = iota 1
||| ```
||| is the same as
||| ```
||| x : Tensor [3, 5] S32
||| x = tensor [[0, 1, 2, 3, 4],
|||             [0, 1, 2, 3, 4],
|||             [0, 1, 2, 3, 4]]
||| ```
||| and
||| ```
||| x : Tensor [3, 5] S32
||| x = iota 0
||| ```
||| is the same as
||| ```
||| x : Tensor [3, 5] S32
||| x = tensor [[0, 0, 0, 0, 0],
|||             [1, 1, 1, 1, 1],
|||             [2, 2, 2, 2, 2]]
||| ```
export
iota : Primitive.Num dtype =>
       {shape : _} ->
       (axis : Nat) ->
       {auto 0 inBounds : InBounds axis shape} ->
       Tensor shape dtype
iota dimension = MkTensor $ Iota shape {dtype} dimension

----------------------------- generic operations ----------------------------

||| Lift a unary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```
||| recip : Tensor [] F64 -> Graph $ Tensor [] F64
||| recip x = pure $ 1.0 / x
||| ```
||| can be lifted to an element-wise reciprocal function as `map recip (tensor [-2, 0.4])`,
||| which produces `tensor [-0.5, 2.5]`.
|||
||| **Note:** Values shared in the same scope as `map` cannot then be used within the scalar
||| function passed to `map`. This is due to an [issue in XLA](https://github.com/openxla/xla/issues/14299).
export
map : (Primitive a, Primitive b) =>
      (Tensor [] a -> Graph $ Tensor [] b) ->
      Tensor shape a -> Graph $ Tensor shape b
map f $ MkTensor {shape = _} x = MkGraphT $ do
  addr <- reserve
  let MkGraphT app = f (MkTensor $ Var addr)
      (env, MkTensor res) = runState (emptyFrom !get) app
      g = MkFn [(addr, MkParameter [] a)] res env

  updateCounterFrom env
  pure $ MkTensor $ Map g [x] (range $ length shape)

||| Lift a binary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```
||| addRecip : Tensor [] F64 -> Tensor [] F64 -> Graph $ Tensor [] F64
||| addRecip x y = pure $ x + 1.0 / y
||| ```
||| can be lifted to an element-wise function as
||| `map2 addRecip (tensor [3.0, -3.0]) (tensor [-2.0, 0.4])`, which produces `tensor [2.5, -0.5]`.
|||
||| **Note:** Values shared in the same scope as `map2` cannot then be used within the scalar
||| function passed to `map2`. This is due to an [issue in XLA](https://github.com/openxla/xla/issues/14299).
export
map2 :
  (Primitive a, Primitive b, Primitive c) =>
  (Tensor [] a -> Tensor [] b -> Graph $ Tensor [] c) ->
  Tensor shape a -> Tensor shape b -> Graph $ Tensor shape c
map2 f (MkTensor {shape = _} x) (MkTensor x') = MkGraphT $ do
  addr0 <- reserve
  addr1 <- reserve
  let MkGraphT app = f (MkTensor $ Var addr0) (MkTensor $ Var addr1)
      (env, MkTensor res) = runState (emptyFrom !get) app
      g = MkFn [(addr0, MkParameter [] a), (addr1, MkParameter [] b)] res env

  updateCounterFrom env
  pure $ MkTensor $ Map g [x, x'] (range $ length shape)

||| Reduce elements along one `axis` of a `Tensor` according to a specified `reducer` `Monoid`.
||| For example, if `x = tensor [[0, 1, 2], [3, 4, 5]]`, then reduce @{Sum} 0 x` produces
||| `tensor [3, 5, 7]`, and `reduce @{Sum} 1 x` produces `tensor [3, 12]`.
|||
||| **Note:** `Semigroup` doesn't use `Graph`, which limits the functions that can be used in
||| `reduce`. However, the most commonly used semigroups don't need `Graph`, including `Sum`,
||| `Prod`, `Min` and `Max`, so for ergonomics, we have opted to use `Monoid` as is. We can
||| provide an overloaded variant if requested.
|||
||| **Note:** Values shared in the same scope as `reduce` cannot then be used within the binary
||| function supplied by the `Monoid`. This is due to an [issue in XLA](https://github.com/openxla/xla/issues/14299).
|||
||| @reducer How to reduce elements along the given `axis`.
||| @axis The axis along which to reduce elements.
export
reduce :
  (reducer : Monoid (Tensor [] dtype)) =>
  Primitive dtype =>
  (axes : List Nat) ->
  {auto 0 axesUnique : Sorted LT axes} ->
  {auto 0 axesInBounds : All (flip InBounds shape) axes} ->
  Tensor shape dtype ->
  Graph $ Tensor (deleteAt axes shape) dtype
reduce axes $ MkTensor x = MkGraphT $ do
  let semigroup : Monoid a -> Semigroup a
      semigroup _ = %search

  addr0 <- reserve
  addr1 <- reserve

  let MkTensor res := (<+>) @{semigroup reducer} (MkTensor $ Var addr0) (MkTensor $ Var addr1)
      g = MkFn [(addr0, MkParameter [] dtype), (addr1, MkParameter [] dtype)] res empty
      MkTensor neutral' = neutral @{reducer}

  pure $ MkTensor $ Reduce g neutral' axes x

||| Sort the elements of a `Tensor` along a specified `dimension` according to a scalar-wise
||| ordering. For sorting function `f`, elements are sorted such that for consecutive sorted
||| elements `a` and `b`, either `f a b` is true, or `f a b` *and* `f b a` are false.
|||
||| **Note:** Sorting is not stable, meaning elements that compare equal according the ordering may
||| be sorted in a different order to the order they appear in the input.
|||
||| **Note:** `sort` is limited to use comparison function without `Graph`. However, since the most
||| commonly-used functions, including (>), (<), (>=), and (<=), don't use `Graph`, we have opted to
||| omit it for ergonomics. We can trivially provide an overloaded variant if requested.
|||
||| **Note:** Values shared in the same scope as `sort` cannot then be used within the scalar
||| function passed to `sort`. This is due to an [issue in XLA](https://github.com/openxla/xla/issues/14299).
|||
||| For example, for `x = tensor [[1, 6, 4], [3, 2, 5]]`, `sort (<) 0 x` produces
||| `tensor [[1, 2, 4], [3, 6, 5]]`, while `sort (<) 1 x` produces
||| `tensor [[1, 4, 6], [2, 3, 5]]`.
export
sort :
  Primitive dtype =>
  (Tensor [] dtype -> Tensor [] dtype -> Tensor [] PRED) ->
  (dimension : Nat) ->
  Tensor shape dtype ->
  {auto 0 dimInBounds : InBounds dimension shape} ->
  Graph $ Tensor shape dtype
sort comp dimension $ MkTensor x = MkGraphT $ do
  addr0 <- reserve
  addr1 <- reserve

  let MkTensor res = comp (MkTensor $ Var addr0) (MkTensor $ Var addr1)
      comparator = MkFn [(addr0, MkParameter [] dtype), (addr1, MkParameter [] dtype)] res empty

  pure $ MkTensor $ Sort comparator dimension False [x]

||| Reverse elements along the specified axes. For example, for
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[-2, -1,  0],
|||             [ 1,  2,  3]]
||| ```
||| `reverse [0] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[ 1,  2,  3],
|||             [-2, -1,  0]]
||| ```
||| `reverse [1] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[ 0, -1, -2],
|||             [ 3,  2,  1]]
||| ```
||| and `reverse [0, 1] x` is
||| ```
||| x : Tensor [2, 3] S32
||| x = tensor [[ 3,  2,  1],
|||             [ 0, -1, -2]]
||| ```
|||
||| **Note:** This function requires `axes` is ordered simply so that elements are unique.
||| The ordering itself is irrelevant to the implementation, but ensures uniqueness without using
||| proofs of contradiction that can be difficult for Idris to construct.
export
reverse :
  (axes : List Nat) ->
  {auto 0 axesUnique : Sorted LT axes} ->
  {auto 0 axesInBounds : All (flip InBounds shape) axes} ->
  Tensor shape dtype ->
  Tensor shape dtype
reverse axes $ MkTensor x = MkTensor $ Reverse axes x

----------------------------- numeric operations ----------------------------

binaryRef : BinaryOp -> Tensor s a -> Tensor s a' -> Tensor s a''
binaryRef op (MkTensor x) (MkTensor x') = MkTensor $ BinaryElementwise op x x'

||| Element-wise equality. For example, `tensor [1, 2] /= tensor [1, 3]` is
||| `tensor [True, False]`.
export
(==) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(==) = binaryRef Eq

||| Element-wise inequality. For example, `tensor [1, 2] /= tensor [1, 3]` is
||| `tensor [False, True]`.
export
(/=) : Primitive.Eq dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(/=) = binaryRef Ne

||| Element-wise less than. For example, `tensor [1, 2, 3] < tensor [2, 2, 2]` is
||| `tensor [True, False, False]`.
export
(<) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<) = binaryRef Lt

||| Element-wise greater than. For example, `tensor [1, 2, 3] > tensor [2, 2, 2]` is
||| `tensor [False, False, True]`.
export
(>) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>) = binaryRef Gt

||| Element-wise less than or equal. For example, `tensor [1, 2, 3] <= tensor [2, 2, 2]`
||| is `tensor [True, True, False]`.
export
(<=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(<=) = binaryRef Le

||| Element-wise greater than or equal. For example,
||| `tensor [1, 2, 3] >= tensor [2, 2, 2]` is `tensor [False, True, True]`.
export
(>=) : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape PRED
(>=) = binaryRef Ge

||| Element-wise boolean and. For example,
||| `tensor [True, True, False, False] && tensor [True, False, True, False]` is
||| `tensor [True, False, False, False]`.
export
(&&) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(&&) = binaryRef And

namespace Semigroup
  export
  [All] Semigroup (Tensor shape PRED) where
    (<+>) = (&&)

namespace Monoid
  export
  [All] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.All where
    neutral = fill True

||| Element-wise boolean or. For example,
||| `tensor [True, True, False, False] || tensor [True, False, True, False]` is
||| `tensor [True, True, True, False]`.
export
(||) : Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED
(||) = binaryRef Or

namespace Semigroup
  export
  [Any] Semigroup (Tensor shape PRED) where
    (<+>) = (||)

namespace Monoid
  export
  [Any] {shape : _} -> Monoid (Tensor shape PRED) using Tensor.Semigroup.Any where
    neutral = fill False

unary : UnaryOp -> Tensor s a -> Tensor s a'
unary op $ MkTensor x = MkTensor $ UnaryElementwise op x

||| Element-wise boolean negation. For example, `not (tensor [True, False])` is
||| `tensor [False, True]`.
export
not : Tensor shape PRED -> Tensor shape PRED
not = unary Not

||| Choose elements from two `Tensor`s based on a `Tensor` of predicates. For each element in the
||| predicates, the output will use the corresponding element from `onTrue` if the element is
||| truthy, else the element from `onFalse`. For example, for
||| ```
||| preds : Tensor [3] PRED
||| preds = tensor [False, True, False]
|||
||| onTrue : Tensor [3] S32
||| onTrue = tensor [1, 2, 3]
|||
||| onFalse : Tensor [3] S32
||| onFalse = tensor [4, 5, 6]
||| ```
||| `select preds onTrue onFalse` is `tensor [4, 2, 6]`.
|||
||| @onTrue The elements to choose where the predicate elements are truthy.
||| @onFalse The elements to choose where the predicate elements are falsy.
export
select :
  Primitive dtype =>
  Tensor shape PRED ->
  (onTrue : Tensor shape dtype) ->
  (onFalse : Tensor shape dtype) ->
  Tensor shape dtype
select (MkTensor p) (MkTensor t) (MkTensor f) = MkTensor $ Select p t f

||| Use a scalar predicate to choose which of two functions to evaluate. If the predicate is truthy,
||| evaluate `onTrue` on the corresponding specified argument, otherwise evaluate `onFalse` on the
||| corresponding specified argument. The result of the evaluated function is returned. For example,
||| for
||| ```
||| x : Tensor [2] S32
||| x = tensor [2, -1]
|||
||| y : Tensor [2, 2] S32
||| y = tensor [[5, 6],
|||             [7, 8]]
||| ```
||| `cond (tensor True) (pure . (tensor 2 *)) x (pure . diag) y` produces `tensor [4, -2]` and
||| `cond (tensor False) (pure . (tensor 2 *)) x (pure . diag) y` produces `tensor [5, 8]`.
|||
||| While both functions will be called for the purposes of defining the computation, only one will
||| be evaluated with its specified argument. That is, this function short-circuits.
|||
||| **Note:** Values shared in the same scope as `cond` cannot then be used in either function
||| passed to `cond`. This is due to an [issue in XLA](https://github.com/openxla/xla/issues/14299).
|||
||| @onTrue The function to execute if the predicate is truthy.
||| @onFalse The function to execute if the predicate is falsy.
export
cond :
  (Primitive tt, Primitive ft, Primitive dtype) =>
  {shape, ts, fs : _} ->
  Tensor [] PRED ->
  (onTrue : Tensor ts tt -> Graph $ Tensor shape dtype) -> Tensor ts tt ->
  (onFalse : Tensor fs ft -> Graph $ Tensor shape dtype) -> Tensor fs ft ->
  Graph $ Tensor shape dtype
cond (MkTensor pred) onTrue (MkTensor true) onFalse (MkTensor false) = MkGraphT $ do
  addr <- reserve

  let MkGraphT app = onTrue (MkTensor $ Var addr)
      (env, MkTensor res) = runState (emptyFrom !get) app
      onTrue = MkFn [(addr, MkParameter ts tt)] res env

  updateCounterFrom env
  addr <- reserve

  let MkGraphT app = onFalse (MkTensor $ Var addr)
      (env, MkTensor res) = runState (emptyFrom !get) app
      onFalse = MkFn [(addr, MkParameter fs ft)] res env

  updateCounterFrom env
  pure $ MkTensor $ Cond pred onTrue true onFalse false

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
export infixl 9 @@

namespace Vector
  ||| Vector dot product with a tensor of any rank. The vector dot product is with the first axis of
  ||| the right-hand side tensor. For example `tensor [0, 1, 2] @@ tensor [-1, -3, -1]` is
  ||| `-1`.
  export
  (@@) : Primitive.Num dtype => Tensor [S m] dtype -> Tensor [S m] dtype -> Tensor [] dtype
  (MkTensor x) @@ (MkTensor x') = MkTensor $ Dot x x'

namespace Matrix
  ||| Matrix multiplication with a matrix or vector. Contraction is along the last axis of the first
  ||| and the first axis of the last. For example,
  ||| ```
  ||| x : Tensor [2, 3] S32
  ||| x = tensor [[-1, -2, -3],
  |||             [ 0,  1,  2]]
  |||
  ||| y : Tensor [3, 1] S32
  ||| y = tensor [[4, 0, 5]]
  |||
  ||| z : Tensor [2, 1] S32
  ||| z = x @@ y
  ||| ```
  ||| is
  ||| ```
  ||| z : Tensor [2, 1] S32
  ||| z = tensor [-19, 10]
  ||| ```
  export
  (@@) : (Primitive dtype, Primitive.Num dtype) =>
         Tensor [n, S m] dtype ->
         Tensor (S m :: tl) dtype ->
         {auto 0 vectorTail : length tl `LTE` 1} ->
         Tensor (n :: tl) dtype
  (MkTensor x) @@ (MkTensor x') = MkTensor $ Dot x x'

||| The output shape of a `dotGeneral` operation.
public export
contract : (lBatch, rBatch, lContract, rContract : List Nat) ->
           (ls, rs : Shape) ->
           {auto 0 lInBoundsBatch : All (flip InBounds ls) lBatch} ->
           {auto 0 rInBoundsBatch : All (flip InBounds rs) rBatch} ->
           {auto 0 lInBoundsContract : All (flip InBounds ls) lContract} ->
           {auto 0 rInBoundsContract : All (flip InBounds rs) rContract} ->
           Shape
contract lBatch rBatch lContract rContract ls rs =
  let lResultDims = deleteAt {inBounds = lInBoundsBatch ++ lInBoundsContract}
                             (lBatch ++ lContract) ls
      rResultDims = deleteAt {inBounds = rInBoundsBatch ++ rInBoundsContract}
                             (rBatch ++ rContract) rs
   in multiIndex lBatch ls ++ lResultDims ++ rResultDims

||| Matrix multiplication.
|||
||| This is a much more general version of `(@@)`, in which you can specify any number of batch
||| and contracting axes. Matrix multiplication is done over each contracting axis.
||| The operation is vectorized over batch axes. For each contracting axis on the left-hand
||| operand, there is one contracting axis on the right-hand operand. These can be different axes
||| in each operand. The same is true for each batch axis.
|||
||| For example, we can vectorize over a typical rank-two matrix multiplication as follows: given
||| two inputs tensors
||| ```
||| let x : Tensor [3, 4, 5, 6] F64
|||     y : Tensor [3, 4, 6, 7] F64
||| ```
||| we do
||| ```
||| let z : Tensor [3, 4, 5, 7] F64 = dotGeneral [0, 1] [0, 1] [3] [2] x y
||| ```
||| Here, we vectorized over the first two axes `[0, 1]`, and do standard matrix multiplication
||| over the remaining axes by specifying the axes 3 and 2 respectively as contracting axes. Notice
||| how the batch axes appear once each at the start of the output shape, and the contracting axis
||| disappears. Remaining axes appear in order from left to right.
|||
||| Note this API is somewhat of a quickfix to bring general matrix multiplication to the tensor
|||   API. It is not thoroughly tested. Expect it to change in the future.
export
dotGeneral : (lBatch, rBatch, lContract, rContract : List Nat) ->
             {auto 0 lUnique : unique (lBatch ++ lContract) = True} ->
             {auto 0 rUnique : unique (rBatch ++ rContract) = True} ->
             {auto 0 lInBoundsBatch : All (flip InBounds ls) lBatch} ->
             {auto 0 rInBoundsBatch : All (flip InBounds rs) rBatch} ->
             {auto 0 lInBoundsContract : All (flip InBounds ls) lContract} ->
             {auto 0 rInBoundsContract : All (flip InBounds rs) rContract} ->
             {auto 0 batchDimsEq : multiIndex lBatch ls = multiIndex rBatch rs} ->
             {auto 0 contractDimsEq : multiIndex lContract ls = multiIndex rContract rs} ->
             Tensor ls dtype ->
             Tensor rs dtype ->
             Tensor (contract lBatch rBatch lContract rContract ls rs) dtype
dotGeneral lb rb lc rc (MkTensor x) (MkTensor y) = MkTensor $ DotGeneral lb rb lc rc x y

||| Element-wise addition. For example, `tensor [1, 2] + tensor [3, 4]` is
||| `tensor [4, 6]`.
export
(+) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) = binaryRef Add

namespace Semigroup
  export
  [Sum] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (+)

namespace Monoid
  export
  [Sum] {shape : _} ->
        Prelude.Num a =>
        PrimitiveRW dtype a =>
        Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Sum where
      neutral = fill 0

||| Element-wise negation. For example, `- tensor [1, -2]` is `tensor [-1, 2]`.
export
negate : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype
negate $ MkTensor i = MkTensor $ UnaryElementwise Neg i

||| Element-wise subtraction. For example, `tensor [3, 4] - tensor [4, 2]` is
||| `tensor [-1, 2]`.
export
(-) : Primitive.Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) = binaryRef Sub

||| Element-wise multiplication. For example, `tensor [2, 3] * tensor [4, 5]` is
||| `tensor [8, 15]`.
export
(*) : Primitive.Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(*) = binaryRef Mul

namespace Scalarwise
  ||| Multiplication by a scalar. For example, `tensor 2 * tensor [3, 5]` is
  ||| `tensor [6, 10]`.
  |||
  ||| The RHS is required to be non-scalar simply to avoid ambiguities with element-wise `(*)`.
  export
  (*) : Primitive.Num dtype => Tensor [] dtype -> Tensor (d :: ds) dtype -> Tensor (d :: ds) dtype
  l * r =
    let MkTensor {shape = _ :: _} _ = r
     in broadcast {shapesOK = scalarToAnyOk (d :: ds)} l * r

namespace Semigroup
  export
  [Prod] Primitive.Num dtype => Semigroup (Tensor shape dtype) where
    (<+>) = (*)

namespace Monoid
  export
  [Prod] {shape : _} ->
         Prelude.Num a =>
         PrimitiveRW dtype a =>
         Primitive.Num dtype =>
    Monoid (Tensor shape dtype) using Semigroup.Prod where
      neutral = fill 1

||| Element-wise floating point division. For example, `tensor [2, 3] / tensor [4, 5]` is
||| `tensor [0.5, 0.6]`.
export
(/) : Primitive.Fractional dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(/) = binaryRef Div

namespace Scalarwise
  ||| Floating point division by a scalar. For example, `tensor [3.4, -5.6] / tensor 2` is
  ||| `tensor [1.7, -2.8]`.
  |||
  ||| The LHS is required to be non-scalar simply to avoid ambiguities with element-wise `(/)`.
  export
  (/) : Primitive.Fractional dtype =>
        Tensor (d :: ds) dtype ->
        Tensor [] dtype ->
        Tensor (d :: ds) dtype
  l / r =
    let MkTensor {shape = _ :: _} _ = l
     in l / broadcast {shapesOK = scalarToAnyOk (d :: ds)} r

||| Element-wise division of natural numbers. For example,
||| `div (tensor [Scalar 13, Scalar 8]) [3, 4]` is `tensor [4, 2]`.
export
div : Tensor shape U64 ->
      (denom : Literal shape Nat) ->
      {auto 0 isSucc : All IsSucc denom} ->
      Tensor shape U64
div x y with (x)
  _ | (MkTensor {shape = _} _) = binaryRef Div x (tensor {dtype = U64} y)

||| Element-wise remainder for natural numbers. For example,
||| `rem (tensor [Scalar 13, Scalar 8]) [3, 4]` is `tensor [1, 0]`.
export
rem : Tensor shape U64 ->
      (denom : Literal shape Nat) ->
      {auto 0 isSucc : All IsSucc denom} ->
      Tensor shape U64
rem x y with (x)
  _ | (MkTensor {shape = _} _) = binaryRef Rem x (tensor {dtype = U64} y)

||| The element-wise reciprocal. For example, `recip (tensor [-2, 0, 0.2])`
||| is `tensor [-0.5, nan, 5]`.
export
recip : Tensor shape F64 -> Tensor shape F64
recip = unary Reciprocal

export infixr 9 ^

||| Each element in `base` raised to the power of the corresponding element in `exponent`.
||| example, `tensor [2, 25, -9] ^ tensor [3, -0.5, 0.5]` is `tensor [8, 0.2, nan]`.
|||
||| Note: The behaviour of this function is not well-defined at negative or positive infinity, or
|||   NaN.
|||
||| Note: The first root is used.
export
(^) : Tensor shape F64 -> Tensor shape F64 -> Tensor shape F64
(^) = binaryRef Pow

||| Element-wise absolute value. For example, `abs (tensor [-2, 3])` is `tensor [2, 3]`.
export
abs : Primitive.Abs dtype => Tensor shape dtype -> Tensor shape dtype
abs = unary Abs

||| The element-wise natural exponential. For example, `exp (tensor [-1, 0, 2])` is
||| `tensor [1 / euler, 1, pow euler 2]`.
export
exp : Tensor shape F64 -> Tensor shape F64
exp = unary Exp

||| The element-wise floor function. For example,
||| `floor (tensor [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `tensor [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floor : Tensor shape F64 -> Tensor shape F64
floor = unary Floor

||| The element-wise ceiling function. For example,
||| `ceil (tensor [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `tensor [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceil : Tensor shape F64 -> Tensor shape F64
ceil = unary Ceil

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `log (tensor [1 / euler, 1, euler * euler])` is `tensor [-1, 0, 2]`.
export
log : Tensor shape F64 -> Tensor shape F64
log = unary Log

||| The element-wise logistic function equivalent to `1 / 1 + exp (-x)`.
export
logistic : Tensor shape F64 -> Tensor shape F64
logistic = unary Logistic

||| The element-wise sine.
export
sin : Tensor shape F64 -> Tensor shape F64
sin = unary Sin

||| The element-wise cosine.
export
cos : Tensor shape F64 -> Tensor shape F64
cos = unary Cos

||| The element-wise tangent.
export
tan : Tensor shape F64 -> Tensor shape F64
tan = unary Tan

||| The element-wise inverse sine.
export
asin : Tensor shape F64 -> Tensor shape F64
asin = unary Asin

||| The element-wise inverse cosine.
export
acos : Tensor shape F64 -> Tensor shape F64
acos = unary Acos

||| The element-wise inverse tangent.
export
atan : Tensor shape F64 -> Tensor shape F64
atan = unary Atan

||| The element-wise hyperbolic sine.
export
sinh : Tensor shape F64 -> Tensor shape F64
sinh = unary Sinh

||| The element-wise hyperbolic cosine.
export
cosh : Tensor shape F64 -> Tensor shape F64
cosh = unary Cosh

||| The element-wise hyperbolic tangent.
export
tanh : Tensor shape F64 -> Tensor shape F64
tanh = unary Tanh

||| The element-wise inverse hyperbolic sine.
export
asinh : Tensor shape F64 -> Tensor shape F64
asinh = unary Asinh

||| The element-wise inverse hyperbolic cosine.
export
acosh : Tensor shape F64 -> Tensor shape F64
acosh = unary Acosh

||| The element-wise inverse hyperbolic tangent.
export
atanh : Tensor shape F64 -> Tensor shape F64
atanh = unary Atanh

||| An approximation to the element-wise error function.
export
erf : Tensor shape F64 -> Tensor shape F64
erf = unary Erf

||| The element-wise square. For example, `square (tensor [-2, 0, 3])`
||| is `tensor [4, 0, 9]`.
export
square : Tensor shape F64 -> Tensor shape F64
square = unary Square

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrt (tensor [0, 9])` is `tensor [0, 3]`.
export
sqrt : Tensor shape F64 -> Tensor shape F64
sqrt = unary Sqrt

||| The element-wise minimum of the first argument compared to the second. For example,
||| `min (tensor [-3, -1, 3]) (tensor [-1, 0, 1])` is `tensor [-3, -1, 1]`.
export
min : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
min (MkTensor x) (MkTensor x') = MkTensor $ BinaryElementwise Min x x'

namespace Semigroup
  export
  [Min] {shape : _} -> Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = min

namespace Monoid
  export
  [Min] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Min where
      neutral = fill (1.0 / 0.0)

||| The element-wise maximum of the first argument compared to the second. For example,
||| `max (tensor [-3, -1, 3]) (tensor [-1, 0, 1])` is `tensor [-1, 0, 3]`.
export
max : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
max (MkTensor x) (MkTensor x') = MkTensor $ BinaryElementwise Max x x'

namespace Semigroup
  export
  [Max] Primitive.Ord dtype => Semigroup (Tensor shape dtype) where
    (<+>) = max

namespace Monoid
  export
  [Max] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Tensor shape dtype) using Semigroup.Max where
      neutral = fill (- 1.0 / 0.0)

highlightNan : Primitive.Ord dtype => Bool -> Tensor [S n] dtype -> Graph $ Tensor [S n] dtype
highlightNan minimize x with (x)
  _ | (MkTensor {shape = _} _) = do
    x <- share x
    cond !(reduce @{All} [0] (x == x)) pure x extremizeNan x

    where

    extremizeNan : {n : _} -> Tensor [S n] dtype -> Graph $ Tensor [S n] dtype
    extremizeNan x = do
      x <- share x
      let min' = broadcast $ Types.min @{NonFinite}
          max' = broadcast $ Types.max @{NonFinite}
      pure $ select (if minimize then x == x else x /= x) max' min'

||| The first index of the minimum value in a vector. For example,
||| `argmin (tensor [-1, 3, -2, -2, 3])` produces `tensor 2`. If the vector contains NaN values,
||| `argmin` returns the index of the first NaN.
|||
||| **Note:** `argmin` uses `Graph` to work around what we believe to be an inconsistency in the XLA
||| compiler's handling of NaN. Specifically, we have modified `argmin` to return the first index of
||| the value returned by `reduce @{Min}`.
export
argmin : Primitive.Ord dtype => Tensor [S n] dtype -> Graph $ Tensor [] U64
argmin x = do
  MkTensor x <- highlightNan True x
  pure $ MkTensor $ Argmin {out = U64} 0 x

||| The first index of the maximum value in a vector. For example,
||| `argmax (tensor [-1, 3, -2, -2, 3])` produces `tensor 1`. If the vector contains NaN values,
||| `argmax` returns the index of the first NaN.
|||
||| **Note:** `argmax` uses `Graph` to work around what we believe to be an inconsistency in the XLA
||| compiler's handling of NaN. Specifically, we have modified `argmax` to return the first index of
||| the value returned by `reduce @{Max}`.
export
argmax : Primitive.Ord dtype => Tensor [S n] dtype -> Graph $ Tensor [] U64
argmax x = do
  MkTensor x <- highlightNan False x
  pure $ MkTensor $ Argmax {out = U64} 0 x

---------------------------- other ----------------------------------

||| Cholesky decomposition. Computes the lower triangular matrix `L` from the symmetric, positive
||| semi-definite matrix `X` s.t. `X = L @@ L.T`. Values will be NaN if the input matrix is not
||| positive semi-definite. The remaining matrix components - those not in the lower triangle or
||| diagonal - will always be zero.
export
cholesky : Tensor [S n, S n] F64 -> Tensor [S n, S n] F64
cholesky $ MkTensor x = triangle Lower (MkTensor $ Cholesky x)

export infix 9 |\, \|

namespace Matrix
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor a) |\ (MkTensor b) = MkTensor $ TriangularSolve a b True

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m, n] F64 -> Tensor [m, n] F64
  (MkTensor a) \| (MkTensor b) = MkTensor $ TriangularSolve a b False

namespace Vector
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Tensor [m, m] F64 -> Tensor [m] F64 -> Tensor [m] F64
  a |\ b = let (MkTensor {shape = [_]} _) = b in squeeze (a |\ expand 1 b)

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Tensor [m, m] F64 -> Tensor [m] F64 -> Tensor [m] F64
  a \| b = let (MkTensor {shape = [_]} _) = b in squeeze (a \| expand 1 b)

||| Sum the elements along the diagonal of the input. For example,
||| `trace (tensor [[-1, 5], [1, 4]])` produces `3`.
export
trace : (Primitive.Num dtype, Prelude.Num a) =>
        PrimitiveRW dtype a =>
        Tensor [S n, S n] dtype ->
        Graph $ Tensor [] dtype
trace x with (x)
  _ | MkTensor {shape = [_, _]} _ = reduce @{Sum} [0, 1] $ x * identity

||| A `Rand a` produces a pseudo-random value of type `a` from a `Tensor [1] U64` state.
||| The state is updated each time a new value is generated.
public export 0
Rand : Type -> Type
Rand = StateT (Tensor [1] U64) Graph

inf : Tensor [] F64
inf = fromDouble (1.0 / 0.0)

||| Generate independent and identically distributed (IID) uniform samples bounded element-wise
||| between `bound` and `bound'`.
|||
||| `bound` and `bound'` need not be ordered, and samples will be generated, elementwise, in
||| [min bound bound', max bound bound'). The exception is where the bounds are equal, in which
||| case: if the bounds are finite, samples are generated at the common bound, else samples are NaN.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two uniform samples
||| ```
||| x : Graph $ Tensor [3] F64
||| x = do let key = tensor (Scalar 2)
|||            initialState = tensor [Scalar 0]
|||        rng <- uniform key (fill 0.0) (fill 1.0)
|||        evalStateT initialState [| rng * rng |]
||| ```
|||
||| @key Determines the stream of generated samples.
||| @bound A bound of the samples. See full docstring for details.
||| @bound' A bound of the samples. See full docstring for details.
export
uniform :
  {shape : _} ->
  (key : Tensor [] U64) ->
  (bound, bound' : Tensor shape F64) ->
  Graph $ Rand $ Tensor shape F64
uniform (MkTensor key) bound bound' = do
  minval@(MkTensor iMinval) <- share $ Tensor.min bound bound'
  maxval@(MkTensor iMaxval) <- share $ Tensor.max bound bound'
  let inf = broadcast inf
  pure $ ST $ \(MkTensor state) => do
    MkTensor x <- share $ MkTensor {shape, dtype = F64}
      $ UniformFloatingPoint key state iMinval iMaxval shape
    let state = MkTensor $ GetTupleElement 1 x
        value = MkTensor $ GetTupleElement 0 x
        -- workaround for XLA bug https://github.com/tensorflow/tensorflow/issues/56663
        -- samples between -inf and 0 should be at -inf, but XLA produces nan
        -- similarly, samples in (inf, inf) should be at inf and respectively for -inf
        value = select ((minval == - inf) && (maxval == fill 0)) (-inf) value
        value = select ((minval == inf) && (maxval == inf)) inf value
        value = select ((minval == -inf) && (maxval == -inf)) (-inf) value
    pure (state, value)

||| Generate independent and identically distributed (IID) samples from the standard normal
||| distribution.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two normal samples
||| ```
||| x : Graph $ Tensor [3] F64
||| x = let key = tensor (Scalar 2)
|||         rng = normal key
|||         initialState = tensor [Scalar 0]
|||      in evalStateT initialState [| rng * rng |]
||| ```
|||
||| @key Determines the stream of generated samples.
export
normal : {shape : _} -> (key : Tensor [] U64) -> Rand $ Tensor shape F64
normal $ MkTensor key =
  ST $ \(MkTensor state) => do
    MkTensor x <- share $ MkTensor {shape, dtype = F64} $ NormalFloatingPoint key state shape
    let state = MkTensor $ GetTupleElement 1 x
        value = MkTensor $ GetTupleElement 0 x
    pure (state, value)
