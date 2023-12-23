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
module Compiler.Eval

import Data.Linear
import Data.Linear.LEither
import Data.Array
import Data.Fin
import Control.Monad.Error.Either
import Control.Monad.Trans
import Control.Linear.LIO

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

data Err : Type where

0 ErrIO : Type -> Type
ErrIO a = EitherT Err IO a

0 LErrIO : Type -> Type
LErrIO a = L1 IO (LEither Err a)

0 Cache : Nat -> Type
Cache n = MArray n (Maybe XlaOp)

interpret : XlaBuilder -> List (Nat, Expr) -> {n : Nat} -> Cache n -@ LErrIO (Cache n)
interpret _          []              cache = pure1 $ Right cache
interpret xlaBuilder ((i, x) :: ixs) cache = do
  Right cache <- interpret xlaBuilder ixs cache
    | Left err => pure1 $ Left err
  Right (xlaOp # cache) <- interpretE x cache
    | Left err => ?interpret_err
  let Just i = natToFin i n
        | Nothing => ?interpret_natToFin_err
  pure1 $ Right $ set i (Just xlaOp) cache

  where

  interpretE : Expr -> Cache n -@ LErrIO (CRes XlaOp $ Cache n)
  interpretE (FromLiteral {dtype} lit) cache = do
    xlaOp <- constantLiteral xlaBuilder !(write {dtype} lit)
    pure1 $ Right $ (xlaOp # cache)
  interpretE (Diag x) cache = do
    let Just x = natToFin x n
          | Nothing => ?interpretE_outOfBounds
        (maybeXlaOp # cache) = Core.get x cache
        Just xlaOp = maybeXlaOp
          | Nothing => ?interpretE_xlaOpMissing
    xlaOp <- getMatrixDiagonal xlaOp
    pure1 $ Right (xlaOp # cache)
  interpretE _        _     = ?interpretE_rhs

compile : String -> Nat -> Env -> ErrIO XlaComputation
compile builderName root = do
  xlaBuilder <- mkXlaBuilder builderName
  -- convert all Nat to Fin n here to save the headache later on
  let (max, env) = toList env
  -- consider unsafeAlloc to avoid headache of handling Maybe ... we know it's safe because it's
  -- topologically sorted
  root <- alloc max Nothing (foo xlaBuilder env)
  build xlaBuilder root

  where

  foo : XlaBuilder -> List (Nat, Expr) -> {n : Nat} -> Cache n -@ !* (ErrIO XlaOp)
  foo xlaBuilder env cache = MkBang $ do
    Right cache <- liftIO1 $ interpret xlaBuilder env cache
      | Left err => ?compile_interpretErr
    let Just root = natToFin root n
          | Nothing => ?compile_outOfBounds
        (maybeXlaOp # cache) = get root cache
        Just xlaOp = maybeXlaOp
          | Nothing => ?compile_xlaOpMissing
    discarding cache $ right xlaOp

export
execute : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> ErrIO $ Literal shape a
execute root env = do
  computation <- compile "root" root env
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
