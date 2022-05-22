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
module Compiler.XLA

import Data.Hashable
import Control.Monad.State
import Data.SortedMap

import Compiler.Graph
import Compiler.TensorFlow.Compiler.XLA.Client.XlaBuilder
import Compiler.TensorFlow.Compiler.XLA.Client.XlaComputation
import Compiler.TensorFlow.Compiler.XLA.Shape
import Compiler.TensorFlow.Compiler.XLA.ShapeUtil
import Compiler.TensorFlow.Compiler.XLA.Literal
import Compiler.TensorFlow.Compiler.XLA.XlaData
import Literal
import Types
import Util

public export
data CachingBuilder : Type where
  MkCachingBuilder : XlaBuilder -> SortedMap Bits64 XlaOp -> CachingBuilder

public export
ComputationContext : Type -> Type
ComputationContext = StateT CachingBuilder IO

cacheInsert : CachingBuilder -> Bits64 -> XlaOp -> CachingBuilder
cacheInsert (MkCachingBuilder builder cache) key xlaOp =
  MkCachingBuilder builder (insert key xlaOp cache)

cacheLookup : CachingBuilder -> Bits64 -> Maybe XlaOp
cacheLookup (MkCachingBuilder _ cache) key = lookup key cache

export
cached : Graph -> ComputationContext XlaOp -> ComputationContext XlaOp
cached graph xs = assert_total $ let graphHash = hash graph in do
  builder <- get
  case cacheLookup builder graphHash of
    Just op => pure op
    Nothing => do
      op <- xs
      builder <- get
      put (cacheInsert builder graphHash op)
      pure op

export
build : HasIO io => String -> ComputationContext XlaOp -> io XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  MkCachingBuilder builder _ <- liftIO $ execStateT (MkCachingBuilder builder empty) x
  build builder

export
buildWithSubBuilder :
  String ->
  List (ComputationContext XlaOp) ->
  ComputationContext XlaOp ->
  ComputationContext XlaComputation
buildWithSubBuilder computationName computationArguments computationResult = do
  MkCachingBuilder builder _ <- get
  subBuilder <- createSubBuilder builder computationName
  let cachingSubBuilder = MkCachingBuilder subBuilder empty
      allOps = sequence_ (computationArguments ++ [computationResult])
  MkCachingBuilder subBuilder _ <- liftIO $ execStateT cachingSubBuilder allOps
  build subBuilder

export
opToString : ComputationContext XlaOp -> String
opToString x = unsafePerformIO $ do
  builder <- mkXlaBuilder "toString"
  (MkCachingBuilder builder _, xlaOp) <- runStateT (MkCachingBuilder builder empty) x
  pure $ opToString builder xlaOp

export
parameter : Primitive dtype => Nat -> List Nat -> String -> (Graph, ComputationContext XlaOp)
parameter position shape name =
  let graph = Leaf "parameter" (cast position) shape (typeString {dtype})

      param : ComputationContext XlaOp
      param = do
        MkCachingBuilder builder _ <- get
        xlaShape <- mkShape {dtype} shape
        cached graph $ parameter builder position xlaShape name

   in (graph, param)

export
interface Primitive dtype => LiteralPrimitiveRW dtype ty where
  set : Literal -> List Nat -> ty -> IO ()
  get : Literal -> List Nat -> ty

indexed : {shape : _} -> Literal shape (List Nat)
indexed = go shape []
  where
  concat : Literal [d] (Literal ds a) -> Literal (d :: ds) a
  concat [] = []
  concat (Scalar x :: xs) = x :: concat xs

  go : (shape : Types.Shape) -> List Nat -> Literal shape (List Nat)
  go [] idxs = Scalar idxs
  go (0 :: _) _ = []
  go (S d :: ds) idxs = concat $ map (\i => go ds (snoc idxs i)) (range (S d))

export
toXLA : HasIO io => LiteralPrimitiveRW dtype a => {shape : _} -> Literal shape a -> io Literal
toXLA xs = liftIO $ do
  literal <- allocLiteral {dtype} shape
  sequence_ [| (\idxs => set {dtype} literal idxs) indexed xs |]
  pure literal

export
fromXLA : LiteralPrimitiveRW dtype a => Literal -> {shape : _} -> Literal shape a
fromXLA lit = map (get {dtype} lit) indexed

export
LiteralPrimitiveRW PRED Bool where
  set = set
  get = get

export
LiteralPrimitiveRW F64 Double where
  set = set
  get = get

export
LiteralPrimitiveRW S32 Int where
  set = set
  get = get

export
LiteralPrimitiveRW U32 Nat where
  set lit idx x = Int.set lit idx (cast x)
  get = cast .: Int.get
