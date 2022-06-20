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
module Compiler.Computation

import Control.Monad.State
import Data.List
import Data.SortedMap

import Data.Hashable

import Compiler.Graph
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaComputation
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Types

public export
data CachingBuilder : Type where
  MkCachingBuilder : XlaBuilder -> SortedMap Bits64 (List (Graph, XlaOp)) -> CachingBuilder

public export
Computation : Type -> Type
Computation = StateT CachingBuilder IO

export
cached : Graph -> Computation XlaOp -> Computation XlaOp
cached graph xs = let graphHash = hash graph in do
  builder <- get
  case cacheLookup builder graphHash of
    Just candidates => case find (\e => fst e == graph) candidates of
      Just (_, op) => pure op
      Nothing => runOp xs graphHash graph candidates
    Nothing => runOp xs graphHash graph []

  where
  cacheUpdate : CachingBuilder -> Bits64 -> List (Graph, XlaOp) -> CachingBuilder
  cacheUpdate (MkCachingBuilder builder cache) key graphOps =
    MkCachingBuilder builder (insert key graphOps cache)

  cacheLookup : CachingBuilder -> Bits64 -> Maybe (List (Graph, XlaOp))
  cacheLookup (MkCachingBuilder _ cache) key = lookup key cache

  runOp : Computation XlaOp -> Bits64 -> Graph -> List (Graph, XlaOp) -> Computation XlaOp
  runOp xs key graph graphOps = do
    op <- xs
    builder <- get
    put (cacheUpdate builder key ((graph, op) :: graphOps))
    pure op

export
build : HasIO io => String -> Computation XlaOp -> io XlaComputation
build computationName x = do
  builder <- mkXlaBuilder computationName
  MkCachingBuilder builder _ <- liftIO $ execStateT (MkCachingBuilder builder empty) x
  build builder

export
buildWithSubBuilder :
  String -> List (Computation XlaOp) -> Computation XlaOp -> Computation XlaComputation
buildWithSubBuilder computationName computationArguments computationResult = do
  MkCachingBuilder builder _ <- get
  subBuilder <- createSubBuilder builder computationName
  let cachingSubBuilder = MkCachingBuilder subBuilder empty
      allOps = sequence_ (computationArguments ++ [computationResult])
  MkCachingBuilder subBuilder _ <- liftIO $ execStateT cachingSubBuilder allOps
  build subBuilder

export
opToString : Computation XlaOp -> String
opToString x = unsafePerformIO $ do
  builder <- mkXlaBuilder "toString"
  (MkCachingBuilder builder _, xlaOp) <- runStateT (MkCachingBuilder builder empty) x
  pure $ opToString builder xlaOp

export
parameter : Primitive dtype => Nat -> Types.Shape -> String -> (Graph, Computation XlaOp)
parameter position shape name =
  let graph = Parameter {dtype} shape position

      param : Computation XlaOp
      param = do
        MkCachingBuilder builder _ <- get
        xlaShape <- mkShape {dtype} shape
        cached graph $ parameter builder position xlaShape name

   in (graph, param)
