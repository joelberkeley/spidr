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

import Debug.Trace
import Control.Monad.Error.Either
import public Control.Monad.State
import public Data.List
import public Data.List.Elem
import Data.List.Quantifiers
import Decidable.Equality

import Compiler.Eval
import Compiler.Expr
import Compiler.LiteralRW
import Literal
import public Primitive
import public Types
import public Util

----------------------------- core definitions ----------------------------

||| A symbolic scalar or array.
|||
||| @shape The `Tensor` shape.
||| @dtype The element type.
export
data Tensor : (shape : Shape) -> (dtype : Type) -> Type where
  MkTensor : Nat -> {shape : _} -> Tensor shape dtype

record EnvN where
  constructor MkEnvN
  max : Nat
  env : Env

-- wrap this, once it's working, in a dedicated `Graph` type
public export 0
Graph : Type -> Type
Graph = State EnvN

addNode : Expr -> {shape : _} -> Graph $ Tensor shape dtype
addNode expr = do
  MkEnvN next env <- get
  put (MkEnvN (S next) (insert next expr env))
  pure $ MkTensor next

||| Construct a `Tensor` from `Literal` data.
export
tensor : PrimitiveRW dtype a => {shape : _} -> Literal shape a -> Graph $ Tensor shape dtype
tensor lit = addNode $ FromLiteral {dtype} {shape} lit

namespace F64
  export
  fromDouble : Double -> Graph $ Tensor [] F64
  fromDouble = tensor . Scalar

namespace S32
  export
  fromInteger : Integer -> Graph $ Tensor [] S32
  fromInteger = tensor . Scalar . fromInteger

||| Evaluate a `Tensor`, returning its value as a `Literal`. This function builds and executes the
||| computation graph.
|||
||| This function will execute the graph on GPU if one is found, else it will use the host CPU.
|||
||| **Note:**
||| * Each call to `eval` will rebuild and execute the graph. Similarly, multiple calls to
|||   `eval` on different `Tensor`s in a computation will be treated entirely independently.
|||   `eval` does not store intermediate values. This is a known limitation, and may change in
|||   the future.
||| * `eval` performs logging. You can disable this by adjusting the TensorFlow logging level
|||    with e.g. `export TF_CPP_MIN_LOG_LEVEL=3`.
export partial
eval : PrimitiveRW dtype ty => Graph (Tensor shape dtype) -> IO (Literal shape ty)
eval x = do
  let (MkEnvN _ env, MkTensor n) = runState (MkEnvN 0 empty) x
  putStrLn "got env: \{show env}"
  runEitherT (run {dtype} n $ traceVal env) <&> \case
    Right lit => lit
    Left err => idris_crash (show err)

||| A string representation of an unevaluated `Tensor`, detailing all enqueued Xla operations.
||| Useful for debugging.
export partial
Show (Graph $ Tensor shape dtype) where
  show x = let (MkEnvN _ env, MkTensor n) = runState (MkEnvN 0 empty) x in
               case unsafePerformIO $ runEitherT $ toString n env of
                    Right str => str

||| Bounds for numeric tensors. Will be infinite for floating point types.
export
[NonFinite] Primitive.Ord dtype => Bounded (Graph $ Tensor [] dtype) where
  min = addNode $ MinValue {dtype}
  max = addNode $ MaxValue {dtype}

||| Finite bounds for numeric tensors.
export
[Finite] Primitive.Ord dtype => Bounded (Graph $ Tensor [] dtype) where
  min = addNode $ MinFiniteValue {dtype}
  max = addNode $ MaxFiniteValue {dtype}

||| Cast the element type. For example, `castDtype (tensor {dtype=S32} [1, -2])` is
||| `tensor {dtype=F64} [1.0, -2.0]`.
export
castDtype : Primitive.Integral a => Tensor shape a -> Graph $ Tensor shape F64
castDtype $ MkTensor x = addNode $ ConvertElementType {dtype = F64} x

----------------------------- structural operations ----------------------------

||| Reshape a `Tensor`. For example, `reshape {to=[2, 1]} (tensor [3, 4])` is
||| `tensor [[3], [4]]`. The output can have a different rank to the input.
export
reshape :
  Primitive dtype =>
  {to : _} ->
  {auto 0 sizesEqual : product from = product to} ->
  Tensor from dtype ->
  Graph $ Tensor to dtype
reshape $ MkTensor {shape} i = addNode $ Reshape shape to i

||| Add a dimension of length one at the specified `axis`. The new dimension will be at the
||| specified `axis` in the new `Tensor` (as opposed to the original `Tensor`). For example,
||| `expand 1 $ tensor [[1, 2], [3, 4], [5, 6]]` is `tensor [[[1, 2]], [[3, 4]], [[5, 6]]]`.
export
expand :
  Primitive dtype =>
  (axis : Nat) ->
  {auto 0 inBounds : axis `LTE` length shape} ->
  Tensor shape dtype ->
  Graph $ Tensor (insertAt axis 1 shape) dtype
expand axis $ MkTensor {shape = _} i = addNode $ Reshape shape (insertAt axis 1 shape) i

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
||| ```idris
||| x : Graph $ Tensor [2, 1, 3, 1] S32
||| x = tensor [[[[4], [5], [6]]],
|||             [[[7], [8], [9]]]]
|||
||| y : Graph $ Tensor [2, 1, 3] S32
||| y = squeeze !x
||| ```
||| is
||| ```idris
||| y : Graph $ Tensor [2, 1, 3] S32
||| y = tensor [[[4, 5, 6]],
|||             [[7, 8, 9]]]
||| ```
export
squeeze :
  Primitive dtype =>
  {to : _} ->
  {auto 0 shapesSqueezable : Squeezable from to} ->
  Tensor from dtype ->
  Graph $ Tensor to dtype
squeeze $ MkTensor {shape} i = addNode $ Reshape shape to i

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
all = Slice 0 @{%search} @{reflexive {ty=Nat}} d

||| A `MultiSlice shape` is a valid multi-dimensionsal slice into a tensor with shape `shape`.
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
  slice {shape=(_ :: _)} (Slice {size} _ _ :: xs) = size :: slice xs
  slice {shape=(_ :: _)} (Index _ :: xs) = slice xs
  slice {shape=(_ :: _)} (DynamicSlice _ size :: xs) = size :: slice xs
  slice {shape=(_ :: _)} (DynamicIndex _ :: xs) = slice xs

||| Slice or index `Tensor` axes. Each axis can be sliced or indexed, and this can be done with
||| either static (`Nat`) or dynamic (scalar `U64`) indices.
|||
||| **Static indices**
|||
||| Static indices are `Nat`s. For example, for
||| ```
||| x : Graph $ Tensor [5, 6] S32
||| x = tensor [[ 0,  1,  2,  3,  4,  5],
|||             [ 6,  7,  8,  9, 10, 11],
|||             [12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23],
|||             [24, 25, 26, 27, 28, 29]]
||| ```
||| we can index as `slice [at 1] !x` to get
||| ```
||| x : Graph $ Tensor [6] S32
||| x = tensor [6, 7, 8, 9, 10, 11]
||| ```
||| or we can slice as `slice [2.to 4] !x` to get
||| ```
||| x : Graph $ Tensor [2, 6] S32
||| x = tensor [[12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23]]
||| ```
||| Note that in `2.to 4`, the 2 is inclusive, and the 4 exclusive, so we return indices 2 and 3.
|||
||| **Dynamic indices**
|||
||| Dynamic indices are scalar `U64` values, and the API works slightly differently because we
||| can't know the value of dynamic indices until the graph is executed. For indexing, with scalar
||| `U64` index `i` in `slice [at i] !x`, `i` is clamped to be a valid index into that dimension.
||| For example, for `i = tensor 1`, `slice [at i] !x` is
||| ```
||| x : Graph $ Tensor [6] S32
||| x = tensor [6, 7, 8, 9, 10, 11]
||| ```
||| as in the static case. However, for `i = tensor 10`, `slice [at i] x` returns the last row
||| ```
||| x : Graph $ Tensor [6] S32
||| x = tensor [24, 25, 26, 27, 28, 29]
||| ```
||| We can also slice by specifying a scalar `U64` start index, and a static size, as
||| `slice [i.size 2] !x` with `i = tensor 2` to get
||| ```
||| x : Graph $ Tensor [2, 6] S32
||| x = tensor [[12, 13, 14, 15, 16, 17],
|||             [18, 19, 20, 21, 22, 23]]
||| ```
||| For a given slice `size`, the dynamic start index is clamped such that we always get `size`
||| elements along that axis. For example, `slice [i.size 2] !x` with `i = tensor 4` is
||| ```
||| x : Graph $ Tensor [2, 6] S32
||| x = tensor [[18, 19, 20, 21, 22, 23],
|||             [24, 25, 26, 27, 28, 29]]
||| ```
||| which starts at index 3 rather than index 4.
|||
||| **Mixed static, dynamic, slicing and indexing**
|||
||| Each axis can only be sliced or indexed, and must use only static or dynamic indices. However,
||| across axes, we can mix these four arbitrarily. For example, with `slice [2.to 4, at 1] !x` to
||| get
||| ```
||| x : Graph $ Tensor [2] S32
||| x = tensor [13, 19]
||| ```
||| or with `i = tensor 2` in `slice [at 1, i.size 2] !x` to get
||| ```
||| x : Graph $ Tensor [2] S32
||| x = tensor [7, 8]
||| ```
|||
||| Slices and indices apply to the leading axes of the tensor. For trailing axes omitted from the
||| multi-dimensional slice, the whole of the axis is returned. If we want to slice or index over
||| later axes and retain all indices in a leading axis, we can use the convenience function `all`,
||| as `slice [all, at 3] x` to get
||| ```
||| x : Graph $ Tensor [5] S32
||| x = tensor [[3], [9], [15], [21], [27]]
||| ```
||| This is exactly the same as the more manual `slice [0.to 5, at 3] !x` and
||| `slice [(tensor 0).size 5, at 3] !x`.
|||
||| @at The multi-dimensional slices and indices at which to slice the tensor.
export
slice :
  Primitive dtype =>
  (at : MultiSlice shape) ->
  Tensor shape dtype ->
  Graph $ Tensor (slice at) dtype
{-
slice at $ MkTensor i env = do
  j <- new
  let env = insert j (Slice (mapd start (const 0) at) (mapd stop id at) (replicate (length shape) 1) i) env
  (dynStartsIdxs, env) <- dynStarts [] env at
  k <- new
  let env = insert k (DynamicSlice dynStartsIdxs (mapd size id at) j) env
  env `addNode` Reshape (mapd size id at) (MultiSlice.slice at) k

      where
      mapd :
        ((Nat -> a) -> {d : Nat} -> SliceOrIndex d -> a) ->
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
      size _ (Slice {size=size'} _ _) = size'
      size _ (Index _) = 1
      size _ (DynamicSlice _ size') = size'
      size _ (DynamicIndex _) = 1

      zero : Expr
      zero = FromLiteral {shape=[]} {dtype=U64} 0

      dynStarts : List Nat -> Env -> {shape : _} -> MultiSlice shape -> Graph (List Nat, Env)
      dynStarts idxs env {shape} [] = f (length shape) (idxs, env)
        where
        f : Nat -> (List Nat, Env) -> Graph (List Nat, Env)
        f 0 (idxs, env) = pure (idxs, env)
        f (S k) (idxs, env) = do
          i <- new
          f k (i :: idxs, insert i zero env)
      dynStarts idxs env (DynamicSlice (MkTensor i env') _ :: ds) = do
        (idxs, env) <- dynStarts idxs env ds
        pure (i :: idxs, mergeLeft env env')
      dynStarts idxs env (DynamicIndex (MkTensor i env') :: ds) = do
        (idxs, env) <- dynStarts idxs env ds
        pure (i :: idxs, mergeLeft env env')
      dynStarts idxs env (_ :: ds) = do
        (idxs, env) <- dynStarts idxs env ds
        i <- new
        pure (i :: idxs, insert i zero env)
-}

||| Concatenate two `Tensor`s along the specfied `axis`. For example,
||| `concat 0 !(tensor [[1, 2], [3, 4]]) !(tensor [[5, 6]])` and
||| `concat 1 !(tensor [[3], [6]]) !(tensor [[4, 5], [7, 8]])` are both
||| `tensor [[1, 2], [3, 4], [5, 6]]`.
export
concat :
  Primitive dtype =>
  (axis : Nat) ->
  Tensor s dtype ->
  Tensor s' dtype ->
  {auto 0 inBounds : (InBounds axis s, InBounds axis s')} ->
  {auto 0 shapesConcatenable : deleteAt axis s = deleteAt axis s'} ->
  Graph $ Tensor (replaceAt axis (index axis s + index axis s') s) dtype
concat axis (MkTensor x) (MkTensor x') = addNode $ Concat axis x x'

||| The diagonal of a matrix as a vector. For example, for
||| ```
||| x : Graph $ Tensor [3, 3] S32
||| x = tensor [[0, 1, 2],
|||             [3, 4, 5],
|||             [6, 7, 8]]
||| ```
||| `diag !x` is `tensor [0, 4, 8]`.
export
diag : Primitive dtype => Tensor [n, n] dtype -> Graph (Tensor [n] dtype)
diag $ MkTensor x = addNode $ Diag x

||| Represents the upper- or lower-trinagular component of a matrix.
public export
data Triangle = Upper | Lower

||| Get the upper- or lower-triangular component of a matrix. For example, for
||| ```
||| x : Graph $ Tensor [3, 3] S32
||| x = tensor [[1, 2, 3],
|||             [4, 5, 6],
|||             [7, 8, 9]]
||| ```
||| `triangle Lower !x` is
||| ```
||| x : Graph $ Tensor [3, 3] S32
||| x = tensor [[1, 0, 0],
|||             [4, 5, 0],
|||             [7, 8, 9]]
||| ```
export
triangle : Primitive dtype => Triangle -> Tensor [n, n] dtype -> Graph $ Tensor [n, n] dtype
triangle tri $ MkTensor x = addNode $ Triangle (case tri of Upper => False; Lower => True) x

||| Tranpose a matrix. For example, `(tensor [[1, 2], [3, 4]]).T` is `tensor [[1, 3], [2, 4]]`.
export
(.T) : Graph (Tensor [m, n] dtype) -> Graph (Tensor [n, m] dtype)
x.T = do
  MkTensor x <- x
  addNode $ Transpose [1, 0] x

||| Transpose axes of a tensor. This is a more general version of `(.T)`, in which you can
||| transpose any number of axes in a tensor of arbitrary rank. The i'th axis in the resulting
||| tensor corresponds to the `index i ordering`'th axis in the input tensor. For example, for
||| ```
||| x : Graph $ Tensor [2, 3, 4] S32
||| x = tensor [[[ 0,  1,  2,  3],
|||              [ 4,  5,  6,  7],
|||              [ 8,  9, 10, 11]],
|||             [[12, 13, 14, 15],
|||              [16, 17, 18, 19],
|||              [20, 21, 22, 23]]]
||| ```
||| `transpose [0, 2, 1] !x` is
||| ```
||| x : Graph $ Tensor [2, 4, 3] S32
||| x = tensor [[[ 0,  4,  8],
|||              [ 1,  5,  9],
|||              [ 2,  6, 10],
|||              [ 3,  7, 11]],
|||             [[12, 16, 20],
|||              [13, 17, 21],
|||              [14, 18, 22],
|||              [15, 19, 23]]]
||| ```
||| `transpose [2, 0, 1] !x` is
||| ```
||| x : Graph $ Tensor [4, 2, 3] S32
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
||| * if an element can be found with `slice [at 3, at 4, at 5] !x` in the original tensor,
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
  {auto 0 unique : Sorted Neq ordering} ->
  {auto 0 inBounds : All (flip InBounds shape) ordering} ->
  Graph $ Tensor (map (dflip List.index shape) ordering) dtype
transpose ordering $ MkTensor x = addNode $ Transpose ordering x

||| The identity tensor, with inferred shape and element type. For example,
||| ```
||| x : Graph $ Tensor [2, 2] S32
||| x = identity
||| ```
||| is
||| ```
||| x : Graph $ Tensor [2, 2] S32
||| x = tensor [[1, 0],
|||             [0, 1]]
||| ```
export
identity : Primitive.Num dtype => {n : _} -> Graph $ Tensor [n, n] dtype
identity = addNode $ Identity {dtype} n

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

||| Broadcast a `Tensor` to a new compatible shape. For example,
|||
||| ```idris
||| x : Graph $ Tensor [2, 3] S32
||| x = broadcast !(tensor [4, 5, 6])
||| ```
|||
||| is
|||
||| ```idris
||| x : Graph $ Tensor [2, 3] S32
||| x = tensor [[4, 5, 6], [4, 5, 6]]
||| ```
export
broadcast :
  Primitive dtype =>
  {to : _} ->
  {auto shapesOK : Broadcastable from to} ->
  Tensor from dtype ->
  Graph $ Tensor to dtype
broadcast $ MkTensor {shape = _} i = addNode $ Broadcast {dtype} from to i

%hint
export
scalarToAnyOk : (to : Shape) -> Broadcastable [] to
scalarToAnyOk [] = Same
scalarToAnyOk (_ :: xs) = Nest (scalarToAnyOk xs)

||| A `Tensor` where every element has the specified value. For example,
|||
||| ```idris
||| fives : Graph $ Tensor [2, 3] S32
||| fives = fill 5
||| ```
||| is
||| ```idris
||| fives : Graph $ Tensor [2, 3] S32
||| fives = tensor [[5, 5, 5],
|||                 [5, 5, 5]]
||| ```
export
fill : PrimitiveRW dtype ty => {shape : _} -> ty -> Graph $ Tensor shape dtype
fill xs = broadcast {shapesOK=scalarToAnyOk shape} !(tensor (Scalar xs))

----------------------------- generic operations ----------------------------

||| Lift a unary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| recip : Tensor [] F64 -> Graph (Tensor [] F64)
||| recip x = 1.0 / pure x
||| ```
||| can be lifted to an element-wise reciprocal function as `map recip !(tensor [-2, 0.4])`,
||| which is `tensor [-0.5, 2.5]`.
export
map :
  (Primitive a, Primitive b) =>
  (Tensor [] a -> Graph $ Tensor [] b) ->
  Tensor shape a ->
  Graph $ Tensor shape b
map f $ MkTensor {shape = _} x = do
  MkEnvN next env <- get
  let params = [(next, MkShapeAndType shape a)]
      subEnv = MkEnvN (S next) (singleton next (Arg next))
      (MkEnvN next subEnv, MkTensor result) = runState subEnv (f $ MkTensor next)
  put (MkEnvN next env)
  addNode $ Map (MkFn params result subEnv) [x] (range $ length shape)

||| Lift a binary function on scalars to an element-wise function on `Tensor`s of arbitrary shape.
||| For example,
||| ```idris
||| addRecip : Tensor [] F64 -> Tensor [] F64 -> Graph $ Tensor [] F64
||| addRecip x y = pure x + 1.0 / pure y
||| ```
||| can be lifted to an element-wise function as
||| `map2 addRecip !(tensor [3.0, -3.0]) !(tensor [-2.0, 0.4])`, which is
||| `tensor [2.5, -0.5]`.
export
map2 :
  (Primitive a, Primitive b, Primitive c) =>
  (Tensor [] a -> Tensor [] b -> Graph $ Tensor [] c) ->
  Tensor shape a ->
  Tensor shape b ->
  Graph $ Tensor shape c
{-
map2 f (MkTensor {shape = _} i env) (MkTensor i' env') = do
  (a0, p0) <- arg
  (a1, p1) <- arg
  MkTensor j subEnv <- f a0 a1
  mergeLeft env env' `addNode` Map (MkFn [p0, p1] j subEnv) [i, i'] (range $ length shape)
-}

||| Reduce elements along one `axis` of a `Tensor` according to a specified `reducer` `Monoid`.
||| For example, if `x = tensor [[0, 1, 2], [3, 4, 5]]`, then reduce @{Sum} 0 !x` is
||| `tensor [3, 5, 7]` and `reduce @{Sum} 1 !x` to `tensor [3, 12]`.
|||
||| @reducer How to reduce elements along the given `axis`.
||| @axis The axis along which to reduce elements.
export
reduce :
  (reducer : Monoid (Graph $ Tensor [] dtype)) =>
  Primitive dtype =>
  (axes : List Nat) ->
  {auto 0 axesUnique : Sorted LT axes} ->
  {auto 0 axesInBounds : All (flip InBounds shape) axes} ->
  Tensor shape dtype ->
  Graph $ Tensor (deleteAt axes shape) dtype
{-
reduce axes $ MkTensor i = do
  MkEnvN max env <- get
  let (MkEnvN _ subEnv, MkTensor l) = runState !get (f $ MkTensor $ S max)
      fn = MkFn [(S max, MkShapeAndType shape a)] l subEnv

  (a0, p0) <- arg
  (a1, p1) <- arg
  let semigroupT : Monoid a -> Semigroup a
      semigroupT _ = %search

  let (subEnvN@(MkEnvN _ subEnv), MkTensor j) = runState !get $ (<+>) @{semigroupT reducer} (pure a0) (pure a1)
      (MkEnvN _ neutralEnv, MkTensor k) = runState subEnvN $ neutral @{reducer}
  addNode $ Reduce (MkFn [p0, p1] j subEnv) k axes i
-}

||| Sort the elements of a `Tensor` along a specified `dimension` according to a scalar-wise
||| ordering. For sorting function `f`, elements are sorted such that for consecutive sorted
||| elements `a` and `b`, either `f a b` is true, or `f a b` *and* `f b a` are false.
|||
||| **Note:** Sorting is not stable, meaning elements that compare equal according the ordering may
||| be sorted in a different order to the order they appear in the input.
|||
||| For example, for `x = tensor [[1, 6, 4], [3, 2, 5]]`, `sort (<) 0 !x` is
||| `tensor [[1, 2, 4], [3, 6, 5]]` and `sort (<) 1 !x` is
||| `tensor [[1, 4, 6], [2, 3, 5]]`.
export
sort :
  Primitive dtype =>
  (Graph (Tensor [] dtype) -> Graph (Tensor [] dtype) -> Graph (Tensor [] PRED)) ->
  (dimension : Nat) ->
  Tensor shape dtype ->
  {auto 0 dimInBounds : InBounds dimension shape} ->
  Graph $ Tensor shape dtype
{-
sort comp dimension $ MkTensor i env = do
  (a0, p0) <- arg
  (a1, p1) <- arg
  MkTensor j subEnv <- comp (pure a0) (pure a1)
  env `addNode` Sort (MkFn [p0, p1] j subEnv) dimension False [i]
-}

||| Reverse elements along the specified axes. For example, for
||| ```
||| x : Graph $ Tensor [2, 3] S32
||| x = tensor [[-2, -1,  0],
|||             [ 1,  2,  3]]
||| ```
||| `reverse [0] !x` is
||| ```
||| x : Graph $ Tensor [2, 3] S32
||| x = tensor [[ 1,  2,  3],
|||             [-2, -1,  0]]
||| ```
||| `reverse [1] !x` is
||| ```
||| x : Graph $ Tensor [2, 3] S32
||| x = tensor [[ 0, -1, -2],
|||             [ 3,  2,  1]]
||| ```
||| and `reverse [0, 1] !x` is
||| ```
||| x : Graph $ Tensor [2, 3] S32
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
  Graph $ Tensor shape dtype
reverse axes $ MkTensor x = addNode $ Reverse axes x

----------------------------- numeric operations ----------------------------

binaryRef : BinaryOp -> Graph (Tensor s a) -> Graph (Tensor s a') -> Graph (Tensor s a'')
binaryRef op x x' = do
  MkTensor x <- x
  MkTensor x' <- x'
  addNode $ BinaryElementwise op x x'

||| Element-wise equality. For example, `tensor [1, 2] /= tensor [1, 3]` is
||| `tensor [True, False]`.
export
(==) : Primitive.Eq dtype =>
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape PRED)
(==) = binaryRef Eq

||| Element-wise inequality. For example, `tensor [1, 2] /= tensor [1, 3]` is
||| `tensor [False, True]`.
export
(/=) : Primitive.Eq dtype =>
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape PRED)
(/=) = binaryRef Ne

||| Element-wise less than. For example, `tensor [1, 2, 3] < tensor [2, 2, 2]` is
||| `tensor [True, False, False]`.
export
(<) : Primitive.Ord dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape PRED)
(<) = binaryRef Lt

||| Element-wise greater than. For example, `tensor [1, 2, 3] > tensor [2, 2, 2]` is
||| `tensor [False, False, True]`.
export
(>) : Primitive.Ord dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape PRED)
(>) = binaryRef Gt

||| Element-wise less than or equal. For example, `tensor [1, 2, 3] <= tensor [2, 2, 2]`
||| is `tensor [True, True, False]`.
export
(<=) : Primitive.Ord dtype =>
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape PRED)
(<=) = binaryRef Le

||| Element-wise greater than or equal. For example,
||| `tensor [1, 2, 3] >= tensor [2, 2, 2]` is `tensor [False, True, True]`.
export
(>=) : Primitive.Ord dtype =>
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape dtype) ->
       Graph (Tensor shape PRED)
(>=) = binaryRef Ge

||| Element-wise boolean and. For example,
||| `tensor [True, True, False, False] && tensor [True, False, True, False]` is
||| `tensor [True, False, False, False]`.
export
(&&) : Graph (Tensor shape PRED) -> Graph (Tensor shape PRED) -> Graph (Tensor shape PRED)
(&&) = binaryRef And

namespace Semigroup
  export
  [All] Semigroup (Graph $ Tensor shape PRED) where
    (<+>) = (&&)

namespace Monoid
  export
  [All] {shape : _} -> Monoid (Graph $ Tensor shape PRED) using Tensor.Semigroup.All where
    neutral = fill True

||| Element-wise boolean or. For example,
||| `tensor [True, True, False, False] || tensor [True, False, True, False]` is
||| `tensor [True, True, True, False]`.
export
(||) : Graph (Tensor shape PRED) ->
       Graph (Tensor shape PRED) ->
       Graph (Tensor shape PRED)
(||) = binaryRef Or

namespace Semigroup
  export
  [Any] Semigroup (Graph $ Tensor shape PRED) where
    (<+>) = (||)

namespace Monoid
  export
  [Any] {shape : _} -> Monoid (Graph $ Tensor shape PRED) using Tensor.Semigroup.Any where
    neutral = fill False

unary : UnaryOp -> Tensor s a -> Graph $ Tensor s a'
unary op $ MkTensor x = addNode $ UnaryElementwise op x

||| Element-wise boolean negation. For example, `not !(tensor [True, False])` is
||| `tensor [False, True]`.
export
not : Tensor shape PRED -> Graph $ Tensor shape PRED
not = unary Not

||| Choose elements from two `Tensor`s based on a `Tensor` of predicates. For each element in the
||| predicates, the output will use the corresponding element from `onTrue` if the element is
||| truthy, else the element from `onFalse`. For example, for
||| ```
||| preds : Graph $ Tensor [3] PRED
||| preds = tensor [False, True, False]
|||
||| onTrue : Graph $ Tensor [3] S32
||| onTrue = tensor [1, 2, 3]
|||
||| onFalse : Graph $ Tensor [3] S32
||| onFalse = tensor [4, 5, 6]
||| ```
||| `select !preds !onTrue !onFalse` is `tensor [4, 2, 6]`.
|||
||| @onTrue The elements to choose where the predicate elements are truthy.
||| @onFalse The elements to choose where the predicate elements are falsy.
export
select :
  Primitive dtype =>
  Tensor shape PRED ->
  (onTrue : Tensor shape dtype) ->
  (onFalse : Tensor shape dtype) ->
  Graph $ Tensor shape dtype
select (MkTensor p) (MkTensor t) (MkTensor f) = addNode $ Select p t f

||| Use a scalar predicate to choose which of two functions to evaluate. If the predicte is truthy,
||| evaluate `onTrue` on the corresponding specified argument, otherwise evaluate `onFalse` on the
||| corresponding specified argument. The result of the evaluated function is returned. For example,
||| for
||| ```
||| x : Graph $ Tensor [2] S32
||| x = tensor [2, -1]
|||
||| y : Graph $ Tensor [2, 2] S32
||| y = tensor [[5, 6],
|||             [7, 8]]
||| ```
||| `cond !(tensor True) (tensor 2 *) !x diag !y` is `tensor [4, -2]` and
||| `cond !(tensor False) (tensor 2 *) !x diag !y` to `tensor [5, 8]`.
|||
||| While both functions will be called for the purposes of defining the computation, only one will
||| be evaluated with its specified argument. That is, this function short-circuits.
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
{-
cond (MkTensor pred envPred) onTrue (MkTensor true envTrue) onFalse (MkTensor false envFalse) = do
  (aTrue, pTrue) <- arg
  (aFalse, pFalse) <- arg
  MkTensor lTrue subEnvTrue <- onTrue aTrue
  MkTensor lFalse subEnvFalse <- onFalse aFalse
  let env = mergeLeft (mergeLeft envPred envTrue) envFalse
  env `addNode` Cond pred (MkFn [pTrue] lTrue subEnvTrue) true (MkFn [pFalse] lFalse subEnvFalse) false
-}

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

namespace Vector
  ||| Vector dot product with a tensor of any rank. The vector dot product is with the first axis of
  ||| the right-hand side tensor. For example `tensor [0, 1, 2] @@ tensor [-1, -3, -1]` is
  ||| `-1`.
  export
  (@@) : Primitive.Num dtype =>
         Graph (Tensor [S m] dtype) ->
         Graph (Tensor [S m] dtype) ->
         Graph (Tensor [] dtype)
  x @@ x' = do
    MkTensor x <- x
    MkTensor x' <- x'
    addNode $ Dot x x'

namespace Matrix
  ||| Matrix multiplication with a matrix or vector. Contraction is along the last axis of the first
  ||| and the first axis of the last. For example:
  |||
  ||| ```idris
  ||| x : Graph $ Tensor [2, 3] S32
  ||| x = tensor [[-1, -2, -3],
  |||             [ 0,  1,  2]]
  |||
  ||| y : Graph $ Tensor [3, 1] S32
  ||| y = tensor [[4, 0, 5]]
  |||
  ||| z : Graph $ Tensor [2, 1] S32
  ||| z = x @@ y
  ||| ```
  |||
  ||| is
  |||
  ||| ```idris
  ||| z : Graph $ Tensor [2, 1] S32
  ||| z = tensor [-19, 10]
  ||| ```
  export
  (@@) : (Primitive dtype, Primitive.Num dtype) =>
         Graph (Tensor [n, S m] dtype) ->
         Graph (Tensor (S m :: tl) dtype) ->
         {auto 0 vectorTail : length tl `LTE` 1} ->
         Graph (Tensor (n :: tl) dtype)
  x @@ x' = do
    MkTensor x <- x
    MkTensor x' <- x'
    addNode $ Dot x x'

||| Element-wise addition. For example, `tensor [1, 2] + tensor [3, 4]` is
||| `tensor [4, 6]`.
export
(+) : Primitive.Num dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype)
(+) = binaryRef Add

namespace Semigroup
  export
  [Sum] Primitive.Num dtype => Semigroup (Graph $ Tensor shape dtype) where
    x <+> x' = x + x'

namespace Monoid
  export
  [Sum] {shape : _} ->
        Prelude.Num a =>
        PrimitiveRW dtype a =>
        Primitive.Num dtype =>
    Monoid (Graph $ Tensor shape dtype) using Semigroup.Sum where
      neutral = fill 0

||| Element-wise negation. For example, `- tensor [1, -2]` is `tensor [-1, 2]`.
export
negate : Primitive.Neg dtype => Graph (Tensor shape dtype) -> Graph (Tensor shape dtype)
negate x = do
  MkTensor i <- x
  addNode $ UnaryElementwise Neg i

||| Element-wise subtraction. For example, `tensor [3, 4] - tensor [4, 2]` is
||| `tensor [-1, 2]`.
export
(-) : Primitive.Neg dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype)
(-) = binaryRef Sub

||| Element-wise multiplication. For example, `tensor [2, 3] * tensor [4, 5]` is
||| `tensor [8, 15]`.
export
(*) : Primitive.Num dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype)
(*) = binaryRef Mul

namespace Scalarwise
  ||| Multiplication by a scalar. For example, `tensor 2 * tensor [3, 5]` is
  ||| `tensor [6, 10]`.
  |||
  ||| The RHS is required to be non-scalar simply to avoid ambiguities with element-wise `(*)`.
  export
  (*) : Primitive.Num dtype =>
        Graph (Tensor [] dtype) ->
        Graph (Tensor (d :: ds) dtype) ->
        Graph (Tensor (d :: ds) dtype)
  l * r = do
    MkTensor {shape=_ :: _} _ <- r
    broadcast {shapesOK=scalarToAnyOk (d :: ds)} !l * r

namespace Semigroup
  export
  [Prod] Primitive.Num dtype => Semigroup (Graph $ Tensor shape dtype) where
    (<+>) = (*)

namespace Monoid
  export
  [Prod] {shape : _} ->
         Prelude.Num a =>
         PrimitiveRW dtype a =>
         Primitive.Num dtype =>
    Monoid (Graph $ Tensor shape dtype) using Semigroup.Prod where
      neutral = fill 1

||| Element-wise floating point division. For example, `tensor [2, 3] / tensor [4, 5]` is
||| `tensor [0.5, 0.6]`.
export
(/) : Primitive.Fractional dtype =>
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype) ->
      Graph (Tensor shape dtype)
(/) = binaryRef Div

namespace Scalarwise
  ||| Floating point division by a scalar. For example, `tensor [3.4, -5.6] / tensor 2` is
  ||| `tensor [1.7, -2.8]`.
  |||
  ||| The LHS is required to be non-scalar simply to avoid ambiguities with element-wise `(/)`.
  export
  (/) : Primitive.Fractional dtype =>
        Graph (Tensor (d :: ds) dtype) ->
        Graph (Tensor [] dtype) ->
        Graph (Tensor (d :: ds) dtype)
  l / r = do
    MkTensor {shape = _ :: _} _ <- l
    l / broadcast {shapesOK=scalarToAnyOk (d :: ds)} !r

||| Element-wise division of natural numbers. For example,
||| `div !(tensor [Scalar 13, Scalar 8]) [3, 4]` is `tensor [4, 2]`.
export
div : Tensor shape U64 ->
      (denom : Literal shape Nat) ->
      {auto 0 isSucc : All IsSucc denom} ->
      Graph $ Tensor shape U64
div x y with (x)
  _ | (MkTensor {shape = _} _) = binaryRef Div (pure x) (tensor {dtype = U64} y)

||| Element-wise remainder for natural numbers. For example,
||| `rem !(tensor [Scalar 13, Scalar 8]) [3, 4]` is `tensor [1, 0]`.
export
rem : Tensor shape U64 ->
      (denom : Literal shape Nat) ->
      {auto 0 isSucc : All IsSucc denom} ->
      Graph $ Tensor shape U64
rem x y with (x)
  _ | (MkTensor {shape = _} _) = binaryRef Rem (pure x) (tensor {dtype = U64} y)

||| The element-wise reciprocal. For example, `recip !(tensor [-2, 0, 0.2])`
||| is `tensor [-0.5, nan, 5]`.
export
recip : Tensor shape F64 -> Graph $ Tensor shape F64
recip = unary Reciprocal

infixr 9 ^

||| Each element in `base` raised to the power of the corresponding element in `exponent`.
||| example, `tensor [2, 25, -9] ^ tensor [3, -0.5, 0.5]` is `tensor [8, 0.2, nan]`.
|||
||| Note: The behaviour of this function is not well-defined at negative or positive infinity, or
|||   NaN.
|||
||| Note: The first root is used.
export
(^) : Graph (Tensor shape F64) -> Graph (Tensor shape F64) -> Graph (Tensor shape F64)
(^) = binaryRef Pow

||| Element-wise absolute value. For example, `abs !(tensor [-2, 3])` is
||| `tensor [2, 3]`.
export
abs : Primitive.Abs dtype => Tensor shape dtype -> Graph $ Tensor shape dtype
abs = unary Abs

||| The element-wise natural exponential. For example, `exp !(tensor [-1, 0, 2])` is
||| `tensor [1 / euler, 1, pow euler 2]`.
export
exp : Tensor shape F64 -> Graph $ Tensor shape F64
exp = unary Exp

||| The element-wise floor function. For example,
||| `floor !(tensor [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `tensor [-2.0, -2.0, -2.0, -1.0, 1.0, 1.0, 1.0, 1.0]`.
export
floor : Tensor shape F64 -> Graph $ Tensor shape F64
floor = unary Floor

||| The element-wise ceiling function. For example,
||| `ceil !(tensor [-1.6, -1.5, -1.4, -1.0, 1.0, 1.4, 1.5, 1.6])` is
||| `tensor [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0]`.
export
ceil : Tensor shape F64 -> Graph $ Tensor shape F64
ceil = unary Ceil

||| The element-wise natural logarithm. Negative inputs yield NaN output. For example,
||| `log !(tensor [1 / euler, 1, euler * euler])` is `tensor [-1, 0, 2]`.
export
log : Tensor shape F64 -> Graph $ Tensor shape F64
log = unary Log

||| The element-wise logistic function equivalent to `1 / 1 + exp !(-x)`.
export
logistic : Tensor shape F64 -> Graph $ Tensor shape F64
logistic = unary Logistic

||| The element-wise sine.
export
sin : Tensor shape F64 -> Graph $ Tensor shape F64
sin = unary Sin

||| The element-wise cosine.
export
cos : Tensor shape F64 -> Graph $ Tensor shape F64
cos = unary Cos

||| The element-wise tangent.
export
tan : Tensor shape F64 -> Graph $ Tensor shape F64
tan = unary Tan

||| The element-wise inverse sine.
export
asin : Tensor shape F64 -> Graph $ Tensor shape F64
asin = unary Asin

||| The element-wise inverse cosine.
export
acos : Tensor shape F64 -> Graph $ Tensor shape F64
acos = unary Acos

||| The element-wise inverse tangent.
export
atan : Tensor shape F64 -> Graph $ Tensor shape F64
atan = unary Atan

||| The element-wise hyperbolic sine.
export
sinh : Tensor shape F64 -> Graph $ Tensor shape F64
sinh = unary Sinh

||| The element-wise hyperbolic cosine.
export
cosh : Tensor shape F64 -> Graph $ Tensor shape F64
cosh = unary Cosh

||| The element-wise hyperbolic tangent.
export
tanh : Tensor shape F64 -> Graph $ Tensor shape F64
tanh = unary Tanh

||| The element-wise inverse hyperbolic sine.
export
asinh : Tensor shape F64 -> Graph $ Tensor shape F64
asinh = unary Asinh

||| The element-wise inverse hyperbolic cosine.
export
acosh : Tensor shape F64 -> Graph $ Tensor shape F64
acosh = unary Acosh

||| The element-wise inverse hyperbolic tangent.
export
atanh : Tensor shape F64 -> Graph $ Tensor shape F64
atanh = unary Atanh

||| An approximation to the element-wise error function.
export
erf : Tensor shape F64 -> Graph $ Tensor shape F64
erf = unary Erf

||| The element-wise square. For example, `square !(tensor [-2, 0, 3])`
||| is `tensor [4, 0, 9]`.
export
square : Tensor shape F64 -> Graph $ Tensor shape F64
square = unary Square

||| The element-wise square root. The first root is used. Negative inputs yield NaN output.
||| For example, `sqrt !(tensor [0, 9])` is `tensor [0, 3]`.
export
sqrt : Tensor shape F64 -> Graph $ Tensor shape F64
sqrt = unary Sqrt

||| The element-wise minimum of the first argument compared to the second. For example,
||| `min !(tensor [-3, -1, 3]) !(tensor [-1, 0, 1])` is `tensor [-3, -1, 1]`.
export
min : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Graph $ Tensor shape dtype
min (MkTensor {shape = _} i) x'@(MkTensor i') = do
  let x = MkTensor i
      op = addNode $ BinaryElementwise Min i i'
  select !(pure x == pure x) !(select !(pure x' == pure x') !op x') x

namespace Semigroup
  export
  [Min] {shape : _} -> Primitive.Ord dtype => Semigroup (Graph $ Tensor shape dtype) where
    x <+> x' = min !x !x'

namespace Monoid
  export
  [Min] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Graph $ Tensor shape dtype) using Semigroup.Min where
      neutral = fill (1.0 / 0.0)

||| The element-wise maximum of the first argument compared to the second. For example,
||| `max !(tensor [-3, -1, 3]) !(tensor [-1, 0, 1])` is `tensor [-1, 0, 3]`.
export
max : Primitive.Ord dtype => Tensor shape dtype -> Tensor shape dtype -> Graph $ Tensor shape dtype
max (MkTensor {shape = _} i) x'@(MkTensor i') = do
  let x = MkTensor i
      op = addNode $ BinaryElementwise Max i i'
  select !(pure x == pure x) !(select !(pure x' == pure x') !op x') x

namespace Semigroup
  export
  [Max] Primitive.Ord dtype => Semigroup (Graph $ Tensor shape dtype) where
    x <+> x' = max !x !x'

namespace Monoid
  export
  [Max] {shape : _} ->
        PrimitiveRW dtype Double =>
        Primitive.Fractional dtype =>
        Primitive.Ord dtype => 
    Monoid (Graph $ Tensor shape dtype) using Semigroup.Max where
      neutral = fill (- 1.0 / 0.0)

highlightNan : Primitive.Ord dtype => Bool -> Tensor [S n] dtype -> Graph $ Tensor [S n] dtype
highlightNan minimize x with (x)
  _ | (MkTensor {shape = _} _) =
    cond !(reduce @{All} [0] !(pure x == pure x)) pure x extremizeNan x

    where

    extremizeNan : {n : _} -> Tensor [S n] dtype -> Graph $ Tensor [S n] dtype
    extremizeNan x = do
      min' <- broadcast !(Types.min @{NonFinite})
      max' <- broadcast !(Types.max @{NonFinite})
      let x : Graph _ = pure x
      select !(if minimize then x == x else x /= x) max' min'

||| The first index of the minimum value in a vector. For example,
||| `argmin !(tensor [-1, 3, -2, -2, 3])` is `tensor 2`. If the vector contains NaN values,
||| `argmin` returns the index of the first NaN.
export
argmin : Primitive.Ord dtype => Tensor [S n] dtype -> Graph $ Tensor [] U64
argmin x = do
  MkTensor x <- highlightNan True x
  addNode $ Argmin {out=U64} 0 x

||| The first index of the maximum value in a vector. For example,
||| `argmax !(tensor [-1, 3, -2, -2, 3])` is `tensor 1`. If the vector contains NaN values,
||| `argmax` returns the index of the first NaN.
export
argmax : Primitive.Ord dtype => Tensor [S n] dtype -> Graph $ Tensor [] U64
argmax x = do
  MkTensor x <- highlightNan False x
  addNode $ Argmax {out=U64} 0 x

---------------------------- other ----------------------------------

||| Cholesky decomposition. Computes the lower triangular matrix `L` from the symmetric, positive
||| semi-definite matrix `X` s.t. `X = L @@ L.T`. Values will be NaN if the input matrix is not
||| positive semi-definite. The remaining matrix components - those not in the lower triangle or
||| diagonal - will always be zero.
export
cholesky : Tensor [S n, S n] F64 -> Graph $ Tensor [S n, S n] F64
cholesky $ MkTensor x = triangle Lower !(addNode $ Cholesky x)

infix 9 |\, \|

namespace Matrix
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Graph (Tensor [m, m] F64) -> Graph (Tensor [m, n] F64) -> Graph (Tensor [m, n] F64)
  x |\ x' = do
    MkTensor x <- x
    MkTensor x' <- x'
    addNode $ TriangularSolve x x' True

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Graph (Tensor [m, m] F64) -> Graph (Tensor [m, n] F64) -> Graph (Tensor [m, n] F64)
  x \| x' = do
    MkTensor x <- x
    MkTensor x' <- x'
    addNode $ TriangularSolve x x' False

namespace Vector
  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is a lower-triangular matrix.
  ||| `a` is given by the lower-triangular elements of the first argument. Values in the
  ||| upper-triangular part are ignored. If `a` is lower-triangular already,
  ||| this is written `a |\ b`.
  |||
  ||| The operator is shaped like the lower-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(\|)`.
  export
  (|\) : Graph (Tensor [m, m] F64) -> Graph (Tensor [m] F64) -> Graph (Tensor [m] F64)
  a |\ b = do
    MkTensor {shape = [_]} i <- b
    squeeze !(a |\ expand 1 (MkTensor {shape = [m]} i))

  ||| Solve the set of linear equations `a @@ x = b` for `x` where `a` is an upper-triangular
  ||| matrix. `a` is given by the upper-triangular elements of the first argument. Values in the
  ||| lower-triangular part are ignored. If `a` is upper-triangular already, this is written
  ||| `a \| b`.
  |||
  ||| The operator is shaped like the upper-triangular portion of a matrix to signal that it uses
  ||| this portion of its argument. This is in contrast to `(|\)`.
  export
  (\|) : Graph (Tensor [m, m] F64) -> Graph (Tensor [m] F64) -> Graph (Tensor [m] F64)
  a \| b = do
    MkTensor {shape=[_]} i <- b
    squeeze !(a \| expand 1 (MkTensor {shape = [m]} i))

||| Sum the elements along the diagonal of the input. For example,
||| `trace !(tensor [[-1, 5], [1, 4]])` is `3`.
export
trace : (Primitive.Num dtype, Prelude.Num a) =>
        PrimitiveRW dtype a =>
        Tensor [S n, S n] dtype ->
        Graph (Tensor [] dtype)
trace x with (x)
  _ | MkTensor {shape=[_, _]} _ = reduce @{Sum} [0, 1] !(Tensor.(*) (pure x) identity)

{-
||| A `Rand a` produces a pseudo-random value of type `a` from a `Tensor [1] U64` state.
||| The state is updated each time a new value is generated.
public export 0
Rand : Type -> Type
Rand = StateT (Tensor [1] U64) Ref

inf : Graph $ Tensor [] F64
inf = fromDouble (1.0 / 0.0)

||| Generate independent and identically distributed (IID) uniform samples bounded element-wise
||| between `bound` and `bound'`.
|||
||| `bound` and `bound'` need not be ordered, and samples will be generated, elementwise, in
||| [min !bound !bound', max !bound !bound'). The exception is where the bounds are equal, in which
||| case: if the bounds are finite, samples are generated at the common bound, else samples are NaN.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two uniform samples
||| ```
||| x : Graph $ Tensor [3] F64
||| x = do key <- tensor (Scalar 2)
|||        rng <- uniform key !(fill 0.0) !(fill 1.0)
|||        initialState <- tensor [Scalar 0]
|||        evalStateT initialState (do lift $ pure !rng * pure !rng)
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
uniform (MkTensor iKey envKey) bound bound' = do
  minval@(MkTensor iMinval envMinval) <- min bound bound'
  maxval@(MkTensor iMaxval envMaxval) <- max bound bound'
  let inf = broadcast !inf
  let env = mergeLeft (mergeLeft envKey envMinval) envMaxval
  pure $ ST $ \(MkTensor iState envState) => do
    i <- new
    let env = mergeLeft envState env
        env = insert i (UniformFloatingPoint iKey iState iMinval iMaxval shape) env
        state = env `addNode` GetTupleElement 1 i
        value = env `addNode` GetTupleElement 0 i
        -- workaround for XLA bug https://github.com/tensorflow/tensorflow/issues/56663
        -- samples between -inf and 0 should be at -inf, but XLA produces nan
        -- similarly, samples in (inf, inf) should be at inf and respectively for -inf
        value = select !((pure minval == - inf) && (pure maxval == fill 0)) !(- inf) !value
        value = select !((pure minval == inf) && (pure maxval == inf)) !inf !value
        value = select !((pure minval == - inf) && (pure maxval == - inf)) !(- inf) !value
    pure (!state, !value)

||| Generate independent and identically distributed (IID) samples from the standard normal
||| distribution.
|||
||| The generated samples are a deterministic function of the input key and state, but may vary
||| between backends and library versions.
|||
||| Example usage, multiplying two normal samples
||| ```
||| x : Graph $ Tensor [3] F64
||| x = do key <- tensor (Scalar 2)
|||        let rng = normal key
|||        initialState <- tensor [Scalar 0]
|||        evalStateT initialState (do lift $ pure !rng * pure !rng)
||| ```
|||
||| @key Determines the stream of generated samples.
export
normal : {shape : _} -> (key : Tensor [] U64) -> Graph $ Tensor shape F64
normal $ MkTensor iKey envKey =
  ST $ \(MkTensor iState envState) => do
    i <- new
    let env = insert i (NormalFloatingPoint iKey iState shape) $ mergeLeft envKey envState
    state <- env `addNode` GetTupleElement 1 i
    value <- env `addNode` GetTupleElement 0 i
    pure (state, value)
-}