{--
Copyright 2023 Joel Berkeley

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
module Compiler.Transform

import Control.Monad.Either
import Data.SortedMap
import Compiler.Expr
import Compiler.LiteralRW
import Literal
import Primitive
import Types

data Err =
    VmapScalar String
  | IndexErr String

Show Err where
  show (VmapScalar _) = "VmapScalar"
  show (IndexErr msg) = "IndexErr \{msg}"

or : Maybe a -> Lazy a -> a
or (Just a) _ = a
or Nothing a = a

data Value = Const | Var Nat

record Acc where
  constructor MkAcc

  ||| Keys are indices of nodes in the original
  metadata : SortedMap Nat Value

  ||| The resulting program shape
  programShape : ProgramShape

  ||| the resulting graph
  graph : Env

||| Traverse the `program` in sorted order. For each `Expr` in the graph, inspect the nodes it is
||| built from. Each node it is built from either
||| * does not exist in `program`. This means that it comes from the global scope, is therefore
|||   constant with respect to the `vmap` argument, and we simply broadcast the value using the
|||   shape extracted from `programShape`.
||| * exists in `program`, in which case ...
||| If a node is built from only constant nodes, it is also constant.
|||
||| @res A pointer to the return value of the original function.
||| @n The size of the vmap-ed dimension.
||| @param A pointer to the parameter in the `vmap`-ed function.
||| @arg A pointer to the argument to `vmap`.
||| @to The return shape of the function to vmap.
||| @localProgram The program to vmap. We vecotrize the whole of this, so this should not include
|||   the whole global program, just the local program containing all values dependent on the value
|||   we vmap over.
||| @globalProgramShape The shape of the whole global program.
export partial
vmap : (res, n, param, arg : Nat) ->
       (to : Shape) ->
       (localProgram : Env) ->
       (globalProgramShape : ProgramShape) ->
       Ref (ProgramShape, Env, Nat)
vmap res n param arg to localProgram globalProgramShape = do
  foo <- runEitherT $ do
    acc <- recurse (toList localProgram) impl (MkAcc empty empty empty)
    case lookup res acc.metadata `or` idris_crash "\{show res} \{show (keys acc.metadata)}" of
         Var i => pure (acc.programShape, acc.graph, i)
         -- todo what is the program shape here?
         Const => lift new <&> \j => (empty, insert j (Broadcast to (n :: to) res) acc.graph, j)
  case foo of
    Right foo => pure foo
    Left err => idris_crash (show err)

  where

  recurse : List (Nat, Expr) -> ((Nat, Expr) -> Acc -> EitherT Err Ref Acc) -> Acc -> EitherT Err Ref Acc
  recurse Nil _ acc = pure acc
  recurse (x :: xs) f acc = do
    acc <- f x acc
    recurse xs f acc

  constant : Nat -> Expr -> Acc -> EitherT Err Ref Acc
  constant i x acc = pure ({ metadata $= insert i Const , graph $= insert i x } acc)

  binary : Nat -> (Nat -> Nat -> Expr) -> Nat -> Nat -> Acc -> EitherT Err Ref Acc
  binary i f j k acc =
    case (lookup j acc.metadata `or` Const, lookup k acc.metadata `or` Const) of
         (Const, Const) => pure ({ metadata $= insert i Const , graph $= insert i (f j k) } acc)
         (Const, Var k) => do
           l <- lift new
           m <- lift new
           -- we need to be careful to only broadcast each value once per graph. We're not
           -- doing that here
           let from = lookup j globalProgramShape `or` idris_crash "Node \{show j} not in globalProgramShape \{show globalProgramShape}"
               graph = insert l (Broadcast from (n :: from) j) acc.graph
               graph = insert m (f l k) graph
           pure $ { metadata $= insert i (Var m) , graph := graph } acc
         (Var j, Const) => do
           l <- lift new
           m <- lift new
           let from = lookup k globalProgramShape `or` idris_crash "\{show k} \{show (keys globalProgramShape)}"
               graph = insert l (Broadcast from (n :: from) k) acc.graph
               graph = insert m (f j l) graph
           pure $ { metadata $= insert i (Var m) , graph := graph } acc
         (Var j, Var k) => do
           l <- lift new
           pure $ { metadata $= insert i (Var l) , graph $= insert l (f j k) } acc

  impl : (Nat, Expr) -> Acc -> EitherT Err Ref Acc
  impl (i, x@(FromLiteral _)) acc = constant i x acc
  impl (i, Arg j) acc =
    if j == param
      then pure ({ metadata $= insert i (Var arg) } acc)
      else lift new <&> \k => { metadata $= insert i Const, graph $= insert k (Arg j) } acc
  impl (i, Tuple js) acc = ?tuple
  impl (i, GetTupleElement idx j) acc = ?getTupleElement
  impl (i, MinValue {dtype}) acc = ?minValue
  impl (i, MaxValue {dtype}) acc = ?maxValue
  impl (i, MinFiniteValue {dtype}) acc = ?minFiniteValue
  impl (i, MaxFiniteValue {dtype}) acc = ?maxFiniteValue
  impl (i, ConvertElementType {dtype} j) acc = ?convertElementType
  impl (i, Reshape from to j) acc =
    case lookup j acc.metadata `or` Const of
         Const => pure ({ metadata $= insert i Const , graph $= insert i (Reshape from to j) } acc)
         Var k => lift new <&> \l =>
           { metadata $= insert i (Var l) , graph $= insert l (Reshape (n :: from) (n :: to) k) } acc
  impl (i, Slice starts stops strides j) acc = ?slice
  impl (i, DynamicSlice starts sizes j) acc = ?dynamicSlice
  impl (i, Concat axis j k) acc = binary i (Concat (S axis)) j k acc
  impl (i, Diag j) acc =
    case lookup j acc.metadata `or` Const of
         -- is this const case right?
         Const => pure ({ metadata $= insert i Const , graph $= insert i (Diag j) } acc)
         Var k => lift new <&> \l => { metadata $= insert i (Var l) , graph $= insert l (Diag k) } acc
  impl (i, Triangle lower j) acc = ?triangle
  impl (i, Transpose axes j) acc = ?transpose
  impl (i, Identity {dtype} size) acc = ?identity
  impl (i, Broadcast from to j) acc = ?broadcast
  impl (i, Reduce f neutral axes j) acc = ?reduce
  impl (i, Sort f dim stable js) acc = ?sort
  impl (i, Reverse axes j) acc = ?reverse
  impl (i, UnaryElementwise {shape} op j) acc = ?unaryElementwise
  impl (i, BinaryElementwise {shape} op j k) acc = ?binaryElementwise
  impl (i, Argmin {out} axis j) acc = ?argmin
  impl (i, Argmax {out} axis j) acc = ?argmax
  impl (i, Select p t f) acc = ?select
  impl (i, Cond p ft t ff f) acc = ?cond
  impl (i, Dot j k) acc = ?dot
  impl (i, Cholesky j) acc = ?cholesky
  impl (i, TriangularSolve j k lower) acc = ?triangularSolve
  impl (i, UniformFloatingPoint key state min max shape) acc = ?uniformFloatingPoint
  impl (i, NormalFloatingPoint key state shape) acc = ?normalFloatingPoint

{-
||| @res The index of the final result in the full environment
||| @n The size of the extra dimensions we're mapping over.
||| @arg The index of the argument to replace
export covering
vmap : (res, n, arg : Nat) -> (unvmapped : Program) -> Expr -> Ref (Program, Nat)
vmap res n arg unvmapped expr = runStateT empty (impl expr)

  where

  impl : Expr -> StateT Program Ref Nat

  recurse : Shape -> Nat -> StateT Program Ref Nat
  recurse shape j =
    case lookup j unvmapped of
         Just expr => impl expr
         Nothing => do
           i <- lift new
           put $ insert i (Broadcast shape (n :: shape) j) !get
           pure i

  impl (FromLiteral {shape, dtype} lit) = do
    i <- lift new
    j <- lift new
    let env = insert i (FromLiteral {shape, dtype} lit) !get
    put $ insert j (Broadcast shape (n :: shape) i) env
    pure j
  impl (Arg {shape} j) =
    if j == arg then pure res
    else do
      i <- lift new
      k <- lift new
      let env = insert i (Arg {shape} j) !get
      put $ insert k (Broadcast shape (n :: shape) i) env
      pure k
  impl (Tuple js) = ?tuple
  impl (GetTupleElement idx j) = ?getTupleElement
  impl (MinValue {dtype}) = ?minValue
  impl (MaxValue {dtype}) = ?maxValue
  impl (MinFiniteValue {dtype}) = ?minFiniteValue
  impl (MaxFiniteValue {dtype}) = ?maxFiniteValue
  impl (ConvertElementType {dtype} j) = ?convertElementType
  impl (Reshape from to j) = do
    j <- recurse from j
    k <- lift new
    put $ insert k (Reshape (n :: from) (n :: to) j) !get
    pure k
  impl (Slice starts stops strides j) = ?slice
  impl (DynamicSlice starts sizes j) = ?dynamicSlice
  impl (Concat {left, right} axis j k) = do
    j <- recurse left j
    k <- recurse right k
    l <- lift new
    put $ insert l (Concat (S axis) {left = n :: left, right = n :: right} j k) !get
    pure l
  impl (Diag {arg} j) = do
    j <- recurse arg j
    k <- lift new
    put $ insert k (Diag {arg = n :: arg} j) !get
    pure k
  impl (Triangle lower j) = ?triangle
  impl (Transpose axes j) = ?transpose
  impl (Identity {dtype} size) = ?identity
  impl (Broadcast from to j) = do
    j <- recurse from j
    k <- lift new
    put $ insert k (Broadcast (n :: from) (n :: to) j) !get
    pure k
  impl (Reduce f neutral axes j) = ?reduce
  impl (Sort f dim stable js) = ?sort
  impl (Reverse axes j) = ?reverse
  impl (UnaryElementwise {shape} op j) = do
    j <- recurse shape j
    k <- lift new
    put $ insert k (UnaryElementwise {shape = n :: shape} op j) !get
    pure k
  impl (BinaryElementwise {shape} op j k) = do
    j <- recurse shape j
    k <- recurse shape k
    l <- lift new
    put $ insert l (BinaryElementwise {shape = n :: shape} op j k) !get
    pure l
  impl (Argmin {out} axis j) = ?argmin
  impl (Argmax {out} axis j) = ?argmax
  impl (Select p t f) = ?select
  impl (Cond p ft t ff f) = ?cond
  impl (Dot j k) = ?dot
  impl (Cholesky j) = ?cholesky
  impl (TriangularSolve j k lower) = ?triangularSolve
  impl (UniformFloatingPoint key state min max shape) = ?uniformFloatingPoint
  impl (NormalFloatingPoint key state shape) = ?normalFloatingPoint
-}