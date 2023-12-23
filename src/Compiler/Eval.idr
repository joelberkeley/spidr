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
interpret _          []              cache = pure1 (LEither.Right cache)
interpret xlaBuilder ((i, x) :: ixs) cache = do
  cacheOrErr <- interpret xlaBuilder ixs cache
  case cacheOrErr of
    Left err => pure1 $ Left err
    Right cache => do
      cacheOrErr <- interpretE x cache
      case cacheOrErr of
        Left err => ?interpret_err
        Right (xlaOp # cache) => case (natToFin i n) of
          Nothing => ?interpret_natToFin_err
          Just i => pure1 $ Right $ set i (Just xlaOp) cache

  where

  interpretE : Expr -> {n : Nat} -> Cache n -@ LErrIO (CRes XlaOp $ Cache n)
  interpretE (FromLiteral {dtype} lit) cache = do
    xlaOp <- constantLiteral xlaBuilder !(write {dtype} lit)
    pure1 $ Right $ (xlaOp # cache)
  interpretE (Diag x) cache =
    case (natToFin x n) of
      Nothing => ?interpretE_diag_err
      Just x => do
        let 1 (xlaOp # cache) = get x cache
        xlaOp <- liftIO1 $ getMatrixDiagonal xlaOp
        pure1 $ Right $ (xlaOp # cache)
  interpretE  _        _     = ?interpretE_rhs

compile : String -> Nat -> Env -> ErrIO XlaComputation
compile builderName root = do
  xlaBuilder <- mkXlaBuilder builderName
  let (max, env) = toList env  -- convert all Nat to Fin n here to save the headache later on
  root <- alloc max Nothing (foo xlaBuilder env)
  build xlaBuilder root

  where

  foo : XlaBuilder -> List (Nat, Expr) -> {n : Nat} -> Cache n -@ !* (ErrIO XlaOp)
  foo xlaBuilder env cache = MkBang $ do
    let foo : LErrIO (Cache n) = interpret xlaBuilder env cache
    1 cacheOrErr <- run foo
    case cacheOrErr of
      Left err => left err
      Right cache => case (natToFin root n) of
        Nothing => ?compile_rhs_natToFin_err
        Just root => let (maybeXlaOp # cache) = get root cache
                      in case maybeXlaOp of
                           Nothing => ?compile_rhs_noXlaOp_err
                           Just xlaOp => right xlaOp

export
execute : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> ErrIO $ Literal shape a
execute root env = do
  computation <- compile "root" root env
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
